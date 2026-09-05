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

use super::IntrinInjecter;
use crate::ir::prim::GE;
use crate::ir::{Expr, IntImm, PrimExpr, Var};
use crate::target::Target;
use crate::tirx::{AssertStmt, AttrStmt, Bind, For, IfThenElse, SeqStmt, Stmt};
use tvm_ffi::{structural_mutate, Result};

#[test]
fn lower_intrin_restores_constraints_after_recursive_error() -> Result<()> {
    crate::libinfo::load_compiler().expect("failed to load the TVM compiler library");
    let x = Var::new("x", "int32")?;
    let expression: PrimExpr = Expr::from(x.clone()).try_into()?;
    let condition: PrimExpr = GE::new(x.clone(), IntImm::new("int32", 0)?)?.into();
    let bound = Var::new("bound", "int32")?;
    let failure = SeqStmt::new(vec![
        AssertStmt::new(condition.clone(), "ValueError", "nonnegative")?.into(),
        Bind::new(bound.clone(), IntImm::new("int32", 1)?)?.into(),
        Bind::new(bound, IntImm::new("int32", 2)?)?.into(),
    ])?;
    let roots: Vec<Stmt> = vec![
        failure.clone().into(),
        IfThenElse::new(condition.clone(), failure.clone())?.into(),
        AttrStmt::new(
            x.clone(),
            "scope",
            IntImm::new("int32", 1)?,
            failure.clone(),
        )?
        .into(),
        For::new(
            Var::new("i", "int32")?,
            IntImm::new("int32", 0)?,
            IntImm::new("int32", 10)?,
            failure,
        )?
        .into(),
    ];
    for root in roots {
        let mut injecter = IntrinInjecter::new(&Target::new("llvm")?, false)?;
        let before = injecter.analyzer.const_int_bound(&expression)?;
        let result = injecter.with_scope(|injecter| structural_mutate(root, injecter));
        let error = result.err().expect("conflicting pure bindings must fail");
        assert!(error.message().contains("Trying to update var"), "{error}");
        assert!(injecter.constraint_exits.is_empty());
        assert!(injecter.aliases.is_empty());
        let after = injecter.analyzer.const_int_bound(&expression)?;
        assert_eq!(
            (before.min_value, before.max_value),
            (after.min_value, after.max_value)
        );
        assert!(!injecter.analyzer.can_prove(&condition)?);
        injecter.with_constraint(&condition, |injecter| {
            assert!(injecter.analyzer.can_prove(&condition)?);
            Ok(())
        })?;
        assert!(!injecter.analyzer.can_prove(&condition)?);
    }
    Ok(())
}
