module instr_decoder (
    input  wire [63:0] instr,      // 64-bit instruction word
    output wire [9:0]  opcode,     // 10-bit opcode
    output wire [7:0]  rd,         // destination register (8-bit)
    output wire [7:0]  rs0,        // source register 0
    output wire [7:0]  rs1,        // source register 1
    output wire [1:0]  pred,       // predicate specifier (2-bit)
    output wire [5:0]  mods,       // modifiers/flags
    output wire [31:0] imm_sext,   // sign-extended immediate (to 32-bit)
    output wire        imm_is_small// immediate fits in 16-bit unsigned
);

// Field extraction (bit positions chosen for alignment and decoder simplicity)
assign opcode = instr[63:54];          // bits 63:54
assign rd     = instr[53:46];          // bits 53:46
assign rs0    = instr[45:38];          // bits 45:38
assign rs1    = instr[37:30];          // bits 37:30
assign pred   = instr[29:28];          // bits 29:28
assign mods   = instr[27:22];          // bits 27:22
wire [21:0] imm_raw = instr[21:0];     // bits 21:0 (22-bit immediate)

// Sign-extend 22-bit immediate to 32-bit for arithmetic/addressing
assign imm_sext = {{10{imm_raw[21]}}, imm_raw}; // replicate sign bit

// Simple helper: does immediate fit in unsigned 16-bit (quick compiler hint)
assign imm_is_small = (imm_raw[21:16] == 6'b0);

endmodule