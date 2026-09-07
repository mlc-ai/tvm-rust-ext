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

use super::{NoOpRemover, RemoveNoOpOptions};
use crate::ir::prim::{FloorDiv, EQ};
use crate::ir::{Expr, IntImm, PrimExpr, Var};
use crate::tirx::{Bind, IfThenElse, SeqStmt};
use tvm_ffi::{structural_mutate, Result};

#[test]
fn remove_no_op_restores_constraints_after_recursive_error() -> Result<()> {
    crate::libinfo::load_compiler().expect("failed to load the TVM compiler library");
    let mut remover = NoOpRemover::new(RemoveNoOpOptions::default())?;
    let x: PrimExpr = Expr::from(Var::new("x", "int32")?).try_into()?;
    let literal = |value| IntImm::new("int32", value);
    // This condition also enters the derived constraints x >= 6 and x < 8.
    let condition: PrimExpr = EQ::new(FloorDiv::new(x.clone(), literal(2)?)?, literal(3)?)?.into();
    let before = remover.analyzer.const_int_bound(&x)?;
    assert!(!remover.analyzer.can_prove(&condition)?);

    // Conflicting definitions make the real recursive AnalyzerBind call fail.
    let bound = Var::new("bound", "int32")?;
    let body = SeqStmt::new(vec![
        Bind::new(bound.clone(), literal(1)?)?.into(),
        Bind::new(bound, literal(2)?)?.into(),
    ])?;
    let root = IfThenElse::new(condition.clone(), body)?;
    let error = structural_mutate(root, &mut remover)
        .err()
        .expect("conflicting definitions must fail");
    assert!(error.message().contains("Trying to update var"), "{error}");

    let after = remover.analyzer.const_int_bound(&x)?;
    assert_eq!(
        (after.min_value, after.max_value),
        (before.min_value, before.max_value)
    );
    assert!(!remover.analyzer.can_prove(&condition)?);
    // The same analyzer must still be usable for a fresh, unrelated scope.
    let different: PrimExpr = EQ::new(x, literal(20)?)?.into();
    remover.analyzer.with_constraint(&different, || {
        assert!(remover.analyzer.can_prove(&different)?);
        Ok(())
    })?;
    assert!(!remover.analyzer.can_prove(&different)?);
    Ok(())
}
