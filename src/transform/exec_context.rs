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

use std::collections::HashMap;

use tvm_ffi::{Array, Map, Result, String as FfiString};

use super::utils::int_value as integer_value;
use crate::ir::{IntImm, PrimExpr};
use crate::tirx::{ScopeBinding, ScopeKind};

const WARP_SIZE: i64 = 32;
const WARPGROUP_SIZE: i64 = 4;

#[derive(Clone)]
pub(super) struct AxisRange {
    pub(super) extent: PrimExpr,
    pub(super) offset: PrimExpr,
    pub(super) stride: PrimExpr,
}

impl AxisRange {
    fn integer(extent: i64, offset: i64, stride: i64) -> Result<Self> {
        Ok(Self {
            extent: integer(extent)?,
            offset: integer(offset)?,
            stride: integer(stride)?,
        })
    }

    fn intersect(&self, low: i64, high: i64) -> Result<Option<Self>> {
        let (offset, extent, stride) = match self.integers() {
            Some(values) if values.2 > 0 => values,
            _ => return Ok(None),
        };
        let index_low = 0.max(ceil_div(low - offset, stride));
        let index_high = extent.min(floor_div(high - 1 - offset, stride) + 1);
        if index_high <= index_low {
            return Ok(None);
        }
        Self::integer(index_high - index_low, offset + stride * index_low, stride).map(Some)
    }

    fn modulo(&self, modulus: i64, residue: i64) -> Result<Option<Self>> {
        if modulus <= 0 {
            return Ok(None);
        }
        let (offset, extent, stride) = match self.integers() {
            Some(values) if values.2 > 0 => values,
            _ => return Ok(None),
        };
        let residue = normalize_mod(residue, modulus);
        let rhs = normalize_mod(residue - offset, modulus);
        let divisor = gcd(stride.abs(), modulus.abs());
        if rhs % divisor != 0 {
            return Ok(None);
        }
        let reduced_stride = stride / divisor;
        let reduced_rhs = rhs / divisor;
        let reduced_modulus = modulus / divisor;
        let period = reduced_modulus;
        let first = normalize_mod(
            reduced_rhs * modular_inverse(reduced_stride, reduced_modulus),
            reduced_modulus,
        );
        if first >= extent {
            return Ok(None);
        }
        Self::integer(
            (extent - 1 - first) / period + 1,
            offset + stride * first,
            stride * period,
        )
        .map(Some)
    }

    fn integers(&self) -> Option<(i64, i64, i64)> {
        Some((
            integer_value(&self.offset)?,
            integer_value(&self.extent)?,
            integer_value(&self.stride)?,
        ))
    }
}

#[derive(Clone)]
pub(super) struct ActiveSet {
    axes: Vec<(String, AxisRange)>,
}

impl ActiveSet {
    fn new(
        lane_extent: i64,
        warp_extent: i64,
        cta_extent: i64,
        cta_axes: &[(String, i64)],
    ) -> Result<Self> {
        let mut axes = vec![
            ("laneid".into(), AxisRange::integer(lane_extent, 0, 1)?),
            ("warpid".into(), AxisRange::integer(warp_extent, 0, 1)?),
        ];
        if cta_axes.is_empty() {
            axes.push(("cta_id".into(), AxisRange::integer(cta_extent, 0, 1)?));
        } else {
            for (name, extent) in cta_axes {
                axes.push((name.clone(), AxisRange::integer(*extent, 0, 1)?));
            }
        }
        Ok(Self { axes })
    }

    pub(super) fn get(&self, name: &str) -> Option<&AxisRange> {
        self.axes
            .iter()
            .find_map(|(candidate, range)| (candidate == name).then_some(range))
    }

    fn with(&self, name: &str, range: AxisRange) -> Option<Self> {
        let mut result = self.clone();
        let (_, current) = result
            .axes
            .iter_mut()
            .find(|(candidate, _)| candidate == name)?;
        *current = range;
        Some(result)
    }

    fn cta_axes(&self) -> impl Iterator<Item = (&str, &AxisRange)> {
        self.axes
            .iter()
            .filter(|(name, _)| name != "laneid" && name != "warpid")
            .map(|(name, range)| (name.as_str(), range))
    }
}

#[derive(Clone, Default)]
pub(super) struct ExecSplit {
    pub(super) inter: HashMap<String, AxisRange>,
    pub(super) intra: HashMap<String, AxisRange>,
}

#[derive(Clone)]
pub(super) struct ExecContext {
    pub(super) active: ActiveSet,
}

impl ExecContext {
    pub(super) fn at_kernel_entry(
        lane_extent: i64,
        warp_extent: i64,
        cta_extent: i64,
        cta_axes: &[(String, i64)],
    ) -> Result<Self> {
        Ok(Self {
            active: ActiveSet::new(lane_extent, warp_extent, cta_extent, cta_axes)?,
        })
    }

    pub(super) fn with_filter(
        &self,
        binding: ScopeBinding,
        low: i64,
        high: i64,
    ) -> Result<Option<Self>> {
        let active = filter_narrow(&self.active, binding, low, high)?;
        Ok(active.map(|active| Self { active }))
    }

    pub(super) fn with_selector(
        &self,
        binding: ScopeBinding,
        selector: PrimExpr,
    ) -> Result<Option<Self>> {
        if binding != ScopeBinding::WARP_THREAD {
            return Ok(None);
        }
        let range = AxisRange {
            extent: integer(1)?,
            offset: selector,
            stride: integer(1)?,
        };
        Ok(self
            .active
            .with("laneid", range)
            .map(|active| Self { active }))
    }

    pub(super) fn with_axis_filter(&self, axis: &str, low: i64, high: i64) -> Result<Option<Self>> {
        if low >= high {
            return Ok(None);
        }
        let Some(range) = self.active.get(axis) else {
            return Ok(None);
        };
        let Some(range) = range.intersect(low, high)? else {
            return Ok(None);
        };
        Ok(self.active.with(axis, range).map(|active| Self { active }))
    }

    pub(super) fn with_axis_modulo(
        &self,
        axis: &str,
        modulus: i64,
        residue: i64,
    ) -> Result<Option<Self>> {
        let Some(range) = self.active.get(axis) else {
            return Ok(None);
        };
        let Some(range) = range.modulo(modulus, residue)? else {
            return Ok(None);
        };
        Ok(self.active.with(axis, range).map(|active| Self { active }))
    }

    pub(super) fn split(&self, scope: ScopeKind) -> Result<Option<ExecSplit>> {
        scope_switch(&self.active, scope)
    }
}

pub(super) fn encode_split(side: &HashMap<String, AxisRange>) -> Map<FfiString, Array<PrimExpr>> {
    Map::from_iter(side.iter().map(|(name, range)| {
        let values = if integer_value(&range.stride) == Some(1) {
            vec![range.extent.clone(), range.offset.clone()]
        } else {
            vec![
                range.extent.clone(),
                range.offset.clone(),
                range.stride.clone(),
            ]
        };
        (FfiString::from(name.as_str()), Array::new(values))
    }))
}

fn filter_narrow(
    active: &ActiveSet,
    binding: ScopeBinding,
    low: i64,
    high: i64,
) -> Result<Option<ActiveSet>> {
    if low >= high {
        return Ok(None);
    }
    if binding == ScopeBinding::WARP_THREAD {
        return narrow_axis(active, "laneid", low, high);
    }
    if binding == ScopeBinding::CTA_WARP {
        return narrow_axis(active, "warpid", low, high);
    }
    if binding == ScopeBinding::KERNEL_CTA || binding == ScopeBinding::CLUSTER_CTA {
        return narrow_axis(active, "cta_id", low, high);
    }
    if binding == ScopeBinding::CTA_THREAD {
        return narrow_flat_thread_range(active, low, high, false);
    }
    if binding == ScopeBinding::WARPGROUP_THREAD {
        return narrow_flat_thread_range(active, low, high, true);
    }
    if binding == ScopeBinding::CTA_WARPGROUP {
        let Some(warps) = active.get("warpid") else {
            return Ok(None);
        };
        let Some((offset, extent, _)) = warps.integers() else {
            return Ok(None);
        };
        if offset % WARPGROUP_SIZE != 0 || extent % WARPGROUP_SIZE != 0 {
            return Ok(None);
        }
        let outer = AxisRange::integer(extent / WARPGROUP_SIZE, offset / WARPGROUP_SIZE, 1)?;
        let Some(outer) = outer.intersect(low, high)? else {
            return Ok(None);
        };
        let Some((new_offset, new_extent, _)) = outer.integers() else {
            return Ok(None);
        };
        return Ok(active.with(
            "warpid",
            AxisRange::integer(new_extent * WARPGROUP_SIZE, new_offset * WARPGROUP_SIZE, 1)?,
        ));
    }
    if binding == ScopeBinding::WARPGROUP_WARP {
        let Some(warps) = active.get("warpid") else {
            return Ok(None);
        };
        let Some((offset, extent, _)) = warps.integers() else {
            return Ok(None);
        };
        let inner_offset = offset % WARPGROUP_SIZE;
        if extent > WARPGROUP_SIZE - inner_offset {
            return Ok(None);
        }
        let inner = AxisRange::integer(extent, inner_offset, 1)?;
        let Some(inner) = inner.intersect(low, high)? else {
            return Ok(None);
        };
        let Some((new_offset, new_extent, _)) = inner.integers() else {
            return Ok(None);
        };
        return Ok(active.with(
            "warpid",
            AxisRange::integer(
                new_extent,
                (offset / WARPGROUP_SIZE) * WARPGROUP_SIZE + new_offset,
                1,
            )?,
        ));
    }
    Ok(None)
}

fn narrow_axis(active: &ActiveSet, axis: &str, low: i64, high: i64) -> Result<Option<ActiveSet>> {
    let Some(current) = active.get(axis) else {
        return Ok(None);
    };
    let Some(narrowed) = current.intersect(low, high)? else {
        return Ok(None);
    };
    Ok(active.with(axis, narrowed))
}

fn narrow_flat_thread_range(
    active: &ActiveSet,
    low: i64,
    high: i64,
    warpgroup_relative: bool,
) -> Result<Option<ActiveSet>> {
    let (Some(lanes), Some(warps)) = (active.get("laneid"), active.get("warpid")) else {
        return Ok(None);
    };
    if !warpgroup_relative {
        let Some((new_warps, new_lanes)) = narrow_flat_product(warps, lanes, low, high)? else {
            return Ok(None);
        };
        return Ok(active
            .with("laneid", new_lanes)
            .and_then(|active| active.with("warpid", new_warps)));
    }
    let Some((warps_in_group, groups)) = factor_warpid(warps)? else {
        return Ok(None);
    };
    let Some((new_warps, new_lanes)) = narrow_flat_product(&warps_in_group, lanes, low, high)?
    else {
        return Ok(None);
    };
    let Some((group_offset, group_extent, _)) = groups.integers() else {
        return Ok(None);
    };
    if group_extent != 1 {
        if same_integer_range(&new_lanes, lanes) && same_integer_range(&new_warps, &warps_in_group)
        {
            return Ok(Some(active.clone()));
        }
        return Ok(None);
    }
    let Some((warp_offset, warp_extent, _)) = new_warps.integers() else {
        return Ok(None);
    };
    Ok(active.with("laneid", new_lanes).and_then(|active| {
        active.with(
            "warpid",
            AxisRange::integer(warp_extent, group_offset * WARPGROUP_SIZE + warp_offset, 1).ok()?,
        )
    }))
}

fn narrow_flat_product(
    major: &AxisRange,
    lane: &AxisRange,
    low: i64,
    high: i64,
) -> Result<Option<(AxisRange, AxisRange)>> {
    let (
        Some((major_offset, major_extent, major_stride)),
        Some((lane_offset, lane_extent, lane_stride)),
    ) = (major.integers(), lane.integers())
    else {
        return Ok(None);
    };
    if major_extent <= 0 || lane_extent <= 0 || major_stride <= 0 || lane_stride <= 0 {
        return Ok(None);
    }
    let active_min = major_offset * WARP_SIZE + lane_offset;
    let active_max = (major_offset + major_stride * (major_extent - 1)) * WARP_SIZE
        + lane_offset
        + lane_stride * (lane_extent - 1)
        + 1;
    if low <= active_min && active_max <= high {
        return Ok(Some((major.clone(), lane.clone())));
    }
    if major_stride != 1 || lane_stride != 1 {
        return Ok(None);
    }
    let lane_high = lane_offset + lane_extent;
    let major_high = major_offset + major_extent;
    let hit_low = major_offset.max(floor_div(low - lane_high, WARP_SIZE) + 1);
    let hit_high = major_high.min(ceil_div(high - lane_offset, WARP_SIZE));
    if hit_high <= hit_low {
        return Ok(None);
    }
    if hit_high == hit_low + 1 {
        let new_lane_low = lane_offset.max(low - hit_low * WARP_SIZE);
        let new_lane_high = lane_high.min(high - hit_low * WARP_SIZE);
        if new_lane_high <= new_lane_low {
            return Ok(None);
        }
        return Ok(Some((
            AxisRange::integer(1, hit_low, 1)?,
            AxisRange::integer(new_lane_high - new_lane_low, new_lane_low, 1)?,
        )));
    }
    if low <= hit_low * WARP_SIZE + lane_offset && (hit_high - 1) * WARP_SIZE + lane_high <= high {
        return Ok(Some((
            AxisRange::integer(hit_high - hit_low, hit_low, 1)?,
            lane.clone(),
        )));
    }
    Ok(None)
}

fn factor_warpid(warps: &AxisRange) -> Result<Option<(AxisRange, AxisRange)>> {
    let Some((offset, extent, stride)) = warps.integers() else {
        return Ok(None);
    };
    if stride != 1 {
        return Ok(None);
    }
    let inner_offset = offset % WARPGROUP_SIZE;
    let group_offset = offset / WARPGROUP_SIZE;
    if inner_offset == 0 && extent % WARPGROUP_SIZE == 0 {
        return Ok(Some((
            AxisRange::integer(WARPGROUP_SIZE, 0, 1)?,
            AxisRange::integer(extent / WARPGROUP_SIZE, group_offset, 1)?,
        )));
    }
    if extent <= WARPGROUP_SIZE - inner_offset {
        return Ok(Some((
            AxisRange::integer(extent, inner_offset, 1)?,
            AxisRange::integer(1, group_offset, 1)?,
        )));
    }
    Ok(None)
}

fn scope_switch(active: &ActiveSet, scope: ScopeKind) -> Result<Option<ExecSplit>> {
    let (Some(lanes), Some(warps)) = (active.get("laneid"), active.get("warpid")) else {
        return Ok(None);
    };
    let mut result = ExecSplit::default();
    if scope == ScopeKind::THREAD {
        result.inter.insert("laneid".into(), lanes.clone());
        result.inter.insert("warpid".into(), warps.clone());
        add_cta_axes(active, &mut result.inter);
    } else if scope == ScopeKind::WARP {
        result.intra.insert("laneid".into(), lanes.clone());
        result.inter.insert("warpid".into(), warps.clone());
        add_cta_axes(active, &mut result.inter);
    } else if scope == ScopeKind::CTA {
        result.intra.insert("laneid".into(), lanes.clone());
        result.intra.insert("warpid".into(), warps.clone());
        add_cta_axes(active, &mut result.inter);
    } else if scope == ScopeKind::CLUSTER {
        result.intra.insert("laneid".into(), lanes.clone());
        result.intra.insert("warpid".into(), warps.clone());
        add_cta_axes(active, &mut result.intra);
    } else if scope == ScopeKind::WARPGROUP {
        let Some((warps_in_group, groups)) = factor_warpid(warps)? else {
            return Ok(None);
        };
        result.intra.insert("laneid".into(), lanes.clone());
        result.intra.insert("wid_in_wg".into(), warps_in_group);
        result.inter.insert("wgid".into(), groups);
        add_cta_axes(active, &mut result.inter);
    } else {
        return Ok(None);
    }
    Ok(Some(result))
}

fn add_cta_axes(active: &ActiveSet, target: &mut HashMap<String, AxisRange>) {
    if let Some(range) = active.get("cta_id") {
        target.insert("cta_id".into(), range.clone());
        return;
    }
    for (name, range) in active.cta_axes() {
        target.insert(name.into(), range.clone());
    }
}

fn integer(value: i64) -> Result<PrimExpr> {
    Ok(IntImm::new("int64", value)?.into())
}

fn same_integer_range(lhs: &AxisRange, rhs: &AxisRange) -> bool {
    lhs.integers() == rhs.integers()
}

fn floor_div(value: i64, divisor: i64) -> i64 {
    if value >= 0 {
        value / divisor
    } else {
        -((-value + divisor - 1) / divisor)
    }
}

fn ceil_div(value: i64, divisor: i64) -> i64 {
    -floor_div(-value, divisor)
}

fn normalize_mod(value: i64, modulus: i64) -> i64 {
    let result = value % modulus;
    if result < 0 {
        result + modulus
    } else {
        result
    }
}

fn gcd(mut lhs: i64, mut rhs: i64) -> i64 {
    while rhs != 0 {
        (lhs, rhs) = (rhs, lhs % rhs);
    }
    lhs.abs()
}

fn extended_gcd(lhs: i64, rhs: i64) -> (i64, i64, i64) {
    if rhs == 0 {
        return (lhs, 1, 0);
    }
    let (divisor, x, y) = extended_gcd(rhs, lhs % rhs);
    (divisor, y, x - (lhs / rhs) * y)
}

fn modular_inverse(value: i64, modulus: i64) -> i64 {
    let (_, inverse, _) = extended_gcd(normalize_mod(value, modulus), modulus);
    normalize_mod(inverse, modulus)
}
