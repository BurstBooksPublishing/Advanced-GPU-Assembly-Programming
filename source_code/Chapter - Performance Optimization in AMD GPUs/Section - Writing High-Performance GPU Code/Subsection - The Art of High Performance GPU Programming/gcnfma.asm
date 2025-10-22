/* s-registers: s0 = pointer to A, s1 = pointer to B, s2 = N (loop count) */
    s_mov_b32 s3, 0                 // s3 = loop counter
    s_mul_i32 s4, s2, 4             // s4 = byte length if needed

    /* Allocate VGPRs: v0..v3 hold loaded A tile; v4..v7 accumulate */
    v_mov_b32 v4, 0                 // acc0 = 0.0f
    v_mov_b32 v5, 0                 // acc1
    v_mov_b32 v6, 0                 // acc2
    v_mov_b32 v7, 0                 // acc3

loop_top:
    s_cmp_eq_i32 s3, s2
    s_cbranch_scc1 done             // exit if counter == N

    /* Bulk-load 4 scalar floats from A and B: use flat loads (coalesced) */
    flat_load_dword v0, s0, s3, 0   // load A[s3] -> v0  ; (comment: coalesced)
    flat_load_dword v1, s1, s3, 0   // load B[s3] -> v1

    /* Ensure loads are visible before FMA */
    s_waitcnt vmcnt(0)              // wait for vector memory ops

    /* Perform FMA: acc += A * B */
    v_mad_f32 v4, v0, v1, v4        // acc0 = v0 * v1 + acc0
    v_mad_f32 v5, v0, v1, v5        // reuse pattern for multiple accumulators
    v_mad_f32 v6, v0, v1, v6
    v_mad_f32 v7, v0, v1, v7

    /* Increment counter and loop */
    s_add_i32 s3, s3, 1
    s_branch loop_top

done:
    /* Reduce wave-local accumulators into a single scalar (example uses v_add) */
    v_add_f32 v4, v4, v5
    v_add_f32 v6, v6, v7
    v_add_f32 v4, v4, v6

    /* Write-back the result (coalesced store) */
    flat_store_dword v4, s0, 0      // store result back to memory
    s_endpgm                        // end program