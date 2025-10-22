; v0-v3: input VGPRs; v4: accumulator VGPR
; s4: loop counter (SGPR), s5: loop bound
loop_start:
  v_fma_f32 v4, v0, v1, v4      ; VALU: v4 += v0 * v1  (per-lane FMA)
  s_add_u32  s4, s4, 1          ; SALU: scalar loop counter increment
  v_cmp_lt_f32 vcc, v4, v2      ; VALU: compare per-lane, sets VCC (vector)
  s_cmp_lt_u32 scc, s4, s5      ; SALU: compare scalar counter, sets SCC
  s_cbranch_scc loop_start      ; SALU: branch on scalar condition (loops)
; next: potentially move first-lane result to SGPR for writeback
  v_readfirstlane_b32 s6, v4    ; VALU->SGPR read of lane0 to s6
  ; ... store s6 to memory via a scalar store sequence or export