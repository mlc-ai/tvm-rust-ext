/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

use std::collections::{HashMap, VecDeque};

use tvm_ffi::Result;

use crate::analysis::Analyzer;
use crate::ir::StringImm;
use crate::ir::{Call, Expr, IntImm, PrimExpr, PrimType, Var};
use crate::prim::{Add, FloorDiv, FloorMod, Mul, EQ, NE};
use crate::tirx::{IterVar, PrimVar, ScopeBinding, ScopeIdDef};

use super::utils::{get_operator, value_error};

pub(super) type LaunchParams = HashMap<String, IterVar>;

pub(super) struct ScopeIdSet {
    definitions: HashMap<ScopeBinding, ScopeIdDef>,
}

impl ScopeIdSet {
    pub(super) fn verify(definitions: &[ScopeIdDef]) -> Result<Self> {
        let analyzer = Analyzer::new()?;
        let mut resolved = HashMap::new();
        let mut queue = VecDeque::new();

        for definition in definitions {
            validate_preferred_extents(definition)?;
            insert_or_upgrade(&mut resolved, &mut queue, definition.clone(), &analyzer)?;
        }
        if resolved.contains_key(&ScopeBinding::CLUSTER_CTA_PAIR)
            && !resolved.contains_key(&ScopeBinding::CLUSTER_CTA)
        {
            return Err(value_error(
                "cta_id_in_pair requires a cluster-to-cta ScopeIdDef in the same kernel",
            ));
        }

        while let Some(head) = queue.pop_front() {
            if head.is_deferred() {
                continue;
            }
            let snapshot = resolved.values().cloned().collect::<Vec<_>>();
            for definition in snapshot {
                if definition.is_deferred() {
                    continue;
                }
                for candidate in [
                    compose(&head, &definition)?,
                    compose(&definition, &head)?,
                    complement(&head, &definition, &analyzer)?,
                    complement(&definition, &head, &analyzer)?,
                ]
                .into_iter()
                .flatten()
                {
                    insert_or_upgrade(&mut resolved, &mut queue, candidate, &analyzer)?;
                }
            }
        }

        for definition in definitions {
            if definition.is_deferred()
                && resolved
                    .get(&definition.scope)
                    .is_none_or(ScopeIdDef::is_deferred)
            {
                return Err(value_error(
                    "cannot infer the extent of a deferred ScopeIdDef",
                ));
            }
        }
        Ok(Self {
            definitions: resolved,
        })
    }

    pub(super) fn get(&self, binding: ScopeBinding) -> Option<&ScopeIdDef> {
        self.definitions.get(&binding)
    }

    pub(super) fn is_empty(&self) -> bool {
        self.definitions.is_empty()
    }

    pub(super) fn resolve_deferred(&self, definition: &ScopeIdDef) -> Result<ScopeIdDef> {
        if !definition.is_deferred() {
            return Ok(definition.clone());
        }
        let source = self
            .get(definition.scope)
            .ok_or_else(|| value_error("deferred ScopeIdDef was not resolved"))?;
        ScopeIdDef::new(
            definition.def_ids.iter().collect(),
            Some(vec![source.fused_extent()?]),
            definition.scope,
            definition
                .preferred_extents
                .as_ref()
                .map(|values| values.iter().collect()),
        )
    }
}

pub(super) fn resolve_scope_id(
    binding: ScopeBinding,
    dimensions: usize,
    target_kind: &str,
    launch_params: &LaunchParams,
) -> Result<Vec<PrimExpr>> {
    if target_kind != "cuda" {
        return Err(value_error("ScopeIdDef resolution currently requires CUDA"));
    }
    if binding == ScopeBinding::KERNEL_CTA {
        return trivial_resolve(launch_params, "blockIdx.", dimensions, false);
    }
    if binding == ScopeBinding::CLUSTER_CTA {
        return trivial_resolve(launch_params, "clusterCtaIdx.", dimensions, true);
    }
    if binding == ScopeBinding::CTA_THREAD {
        return trivial_resolve(launch_params, "threadIdx.", dimensions, false);
    }
    if binding == ScopeBinding::KERNEL_CLUSTER {
        if dimensions > 3 {
            return Err(value_error(
                "kernel-to-cluster scope supports at most 3 dimensions",
            ));
        }
        let operator = get_operator("tirx.cuda.mov_sreg")?;
        return (0..dimensions)
            .map(|dimension| {
                PrimExpr::try_from(Expr::from(Call::new(
                    PrimType::new("int32")?,
                    operator.clone(),
                    vec![
                        IntImm::new("int32", 32)?.into(),
                        StringImm::new(&format!("clusterid.{}", axis_name(dimension))).into(),
                    ],
                )))
            })
            .collect();
    }

    if dimensions != 1 {
        return Err(value_error(
            "this ScopeIdDef binding must be one-dimensional",
        ));
    }
    let analyzer = Analyzer::new()?;
    let warp = get_thread(launch_params, "warp_id_in_cta", false)?.0;
    let linear_thread = linear_thread_index(launch_params)?;
    let result = if binding == ScopeBinding::CTA_WARPGROUP {
        FloorDiv::new(warp, IntImm::new("int32", 4)?)?.into()
    } else if binding == ScopeBinding::CTA_WARP {
        warp
    } else if binding == ScopeBinding::WARPGROUP_WARP {
        FloorMod::new(warp, IntImm::new("int32", 4)?)?.into()
    } else if binding == ScopeBinding::WARPGROUP_THREAD {
        FloorMod::new(linear_thread, IntImm::new("int32", 128)?)?.into()
    } else if binding == ScopeBinding::WARP_THREAD {
        FloorMod::new(linear_thread, IntImm::new("int32", 32)?)?.into()
    } else if binding == ScopeBinding::CLUSTER_CTA_PAIR {
        let (x, extent_x) = get_thread(launch_params, "clusterCtaIdx.x", true)?;
        let (y, extent_y) = get_thread(launch_params, "clusterCtaIdx.y", true)?;
        let (z, _) = get_thread(launch_params, "clusterCtaIdx.z", true)?;
        let y_term: PrimExpr = Mul::new(y, extent_x.clone())?.into();
        let z_term: PrimExpr = Mul::new(z, Mul::new(extent_x, extent_y)?)?.into();
        let linear: PrimExpr = Add::new(Add::new(x, y_term)?, z_term)?.into();
        FloorMod::new(linear, IntImm::new("int32", 2)?)?.into()
    } else {
        return Err(value_error("unknown ScopeIdDef binding"));
    };
    Ok(vec![analyzer.simplify(&result)?])
}

pub(super) fn compute_warp_id_in_cta(launch_params: &LaunchParams) -> Result<PrimExpr> {
    let linear = linear_thread_index(launch_params)?;
    let warp: PrimExpr = FloorDiv::new(linear, IntImm::new("int32", 32)?)?.into();
    let operator = get_operator("tirx.tvm_warp_shuffle")?;
    PrimExpr::try_from(Expr::from(Call::new(
        warp.type_annotation(),
        operator,
        vec![
            IntImm::new("uint32", 0xffff_ffff)?.into(),
            warp.into(),
            IntImm::new("int32", 0)?.into(),
            IntImm::new("int32", 32)?.into(),
            IntImm::new("int32", 32)?.into(),
        ],
    )))
}

fn insert_or_upgrade(
    definitions: &mut HashMap<ScopeBinding, ScopeIdDef>,
    queue: &mut VecDeque<ScopeIdDef>,
    candidate: ScopeIdDef,
    analyzer: &Analyzer,
) -> Result<()> {
    let Some(existing) = definitions.get(&candidate.scope).cloned() else {
        if !candidate.is_deferred() {
            queue.push_back(candidate.clone());
        }
        definitions.insert(candidate.scope, candidate);
        return Ok(());
    };
    if existing.is_deferred() && !candidate.is_deferred() {
        let upgraded = ScopeIdDef::new(
            existing.def_ids.iter().collect(),
            Some(vec![candidate.fused_extent()?]),
            existing.scope,
            existing
                .preferred_extents
                .as_ref()
                .map(|values| values.iter().collect()),
        )?;
        queue.push_back(upgraded.clone());
        definitions.insert(upgraded.scope, upgraded);
    } else if !existing.is_deferred()
        && !candidate.is_deferred()
        && !analyzer.can_prove_equal(&existing.fused_extent()?, &candidate.fused_extent()?)?
    {
        return Err(value_error("inconsistent ScopeIdDef extents"));
    }
    Ok(())
}

fn validate_preferred_extents(definition: &ScopeIdDef) -> Result<()> {
    let Some(preferred) = &definition.preferred_extents else {
        return Ok(());
    };
    if definition.scope != ScopeBinding::CLUSTER_CTA {
        return Err(value_error(
            "preferred extents are only valid for cluster-to-cta scope",
        ));
    }
    let Some(extents) = &definition.extents else {
        return Err(value_error(
            "preferred extents cannot be attached to a deferred ScopeIdDef",
        ));
    };
    if preferred.len() != extents.len() {
        return Err(value_error(
            "preferred ScopeIdDef extents must match the extent dimensions",
        ));
    }
    Ok(())
}

fn compose(lhs: &ScopeIdDef, rhs: &ScopeIdDef) -> Result<Option<ScopeIdDef>> {
    if lhs.is_deferred()
        || rhs.is_deferred()
        || lhs.scope == ScopeBinding::CLUSTER_CTA_PAIR
        || rhs.scope == ScopeBinding::CLUSTER_CTA_PAIR
    {
        return Ok(None);
    }
    let (lhs_parent, lhs_child) = lhs.scope.name_pair()?;
    let (rhs_parent, rhs_child) = rhs.scope.name_pair()?;
    if lhs_child != rhs_parent {
        return Ok(None);
    }
    let Some(binding) = binding_from_parts(lhs_parent, rhs_child) else {
        return Ok(None);
    };
    generated_definition(
        binding,
        Mul::new(lhs.fused_extent()?, rhs.fused_extent()?)?.into(),
    )
    .map(Some)
}

fn complement(
    lhs: &ScopeIdDef,
    rhs: &ScopeIdDef,
    analyzer: &Analyzer,
) -> Result<Option<ScopeIdDef>> {
    if lhs.is_deferred()
        || rhs.is_deferred()
        || lhs.scope == ScopeBinding::CLUSTER_CTA_PAIR
        || rhs.scope == ScopeBinding::CLUSTER_CTA_PAIR
    {
        return Ok(None);
    }
    let rhs_extent = rhs.fused_extent()?;
    if analyzer.can_prove_equal(&rhs_extent, &IntImm::new("int32", 0)?.into())? {
        return Ok(None);
    }
    let (lhs_parent, lhs_child) = lhs.scope.name_pair()?;
    let (rhs_parent, rhs_child) = rhs.scope.name_pair()?;
    let binding = if lhs_parent == rhs_parent && scope_rank(rhs_child) < scope_rank(lhs_child) {
        binding_from_parts(rhs_child, lhs_child)
    } else if lhs_child == rhs_child && scope_rank(lhs_parent) < scope_rank(rhs_parent) {
        binding_from_parts(lhs_parent, rhs_parent)
    } else {
        None
    };
    let Some(binding) = binding else {
        return Ok(None);
    };
    let lhs_extent = lhs.fused_extent()?;
    let remainder: PrimExpr = FloorMod::new(lhs_extent.clone(), rhs_extent.clone())?.into();
    let zero: PrimExpr = IntImm::new("int32", 0)?.into();
    if analyzer.can_prove(&EQ::new(remainder.clone(), zero.clone())?.into())? {
        return generated_definition(binding, FloorDiv::new(lhs_extent, rhs_extent)?.into())
            .map(Some);
    }
    if analyzer.can_prove(&NE::new(remainder, zero)?.into())? {
        return Err(value_error("ScopeIdDef extents are not divisible"));
    }
    Ok(None)
}

fn generated_definition(binding: ScopeBinding, extent: PrimExpr) -> Result<ScopeIdDef> {
    let variable = PrimVar::try_from(Var::with_type("", extent.type_annotation()))?;
    ScopeIdDef::new(vec![variable], Some(vec![extent]), binding, None)
}

fn trivial_resolve(
    launch_params: &LaunchParams,
    prefix: &str,
    dimensions: usize,
    allow_missing: bool,
) -> Result<Vec<PrimExpr>> {
    (0..dimensions)
        .map(|dimension| {
            get_thread(
                launch_params,
                &format!("{prefix}{}", axis_name(dimension)),
                allow_missing,
            )
            .map(|pair| pair.0)
        })
        .collect()
}

fn get_thread(
    launch_params: &LaunchParams,
    tag: &str,
    allow_missing: bool,
) -> Result<(PrimExpr, PrimExpr)> {
    let Some(variable) = launch_params.get(tag) else {
        if allow_missing {
            return Ok((
                IntImm::new("int32", 0)?.into(),
                IntImm::new("int32", 1)?.into(),
            ));
        }
        return Err(value_error(&format!("cannot find thread variable {tag}")));
    };
    let domain = variable
        .dom()?
        .ok_or_else(|| value_error("launch parameter has no domain"))?;
    Ok((variable.var()?.into(), domain.extent.clone()))
}

fn linear_thread_index(launch_params: &LaunchParams) -> Result<PrimExpr> {
    let (x, extent_x) = get_thread(launch_params, "threadIdx.x", true)?;
    let (y, extent_y) = get_thread(launch_params, "threadIdx.y", true)?;
    let (z, _) = get_thread(launch_params, "threadIdx.z", true)?;
    let y_term: PrimExpr = Mul::new(y, extent_x.clone())?.into();
    let z_term: PrimExpr = Mul::new(z, Mul::new(extent_x, extent_y)?)?.into();
    let expression: PrimExpr = Add::new(Add::new(x, y_term)?, z_term)?.into();
    Analyzer::new()?.simplify(&expression)
}

fn axis_name(dimension: usize) -> char {
    char::from(b'x' + u8::try_from(dimension).expect("scope dimensions are limited to three"))
}

fn binding_from_parts(parent: &str, child: &str) -> Option<ScopeBinding> {
    match (parent, child) {
        ("kernel", "cluster") => Some(ScopeBinding::KERNEL_CLUSTER),
        ("kernel", "cta") => Some(ScopeBinding::KERNEL_CTA),
        ("cluster", "cta") => Some(ScopeBinding::CLUSTER_CTA),
        ("cta", "warpgroup") => Some(ScopeBinding::CTA_WARPGROUP),
        ("cta", "warp") => Some(ScopeBinding::CTA_WARP),
        ("warpgroup", "warp") => Some(ScopeBinding::WARPGROUP_WARP),
        ("warp", "thread") => Some(ScopeBinding::WARP_THREAD),
        ("cta", "thread") => Some(ScopeBinding::CTA_THREAD),
        ("warpgroup", "thread") => Some(ScopeBinding::WARPGROUP_THREAD),
        ("cluster", "cta_pair") => Some(ScopeBinding::CLUSTER_CTA_PAIR),
        _ => None,
    }
}

fn scope_rank(scope: &str) -> i32 {
    match scope {
        "kernel" => -1,
        "cluster" => 2,
        "cta" => 3,
        "warpgroup" => 4,
        "warp" => 5,
        "thread" => 6,
        _ => 7,
    }
}
