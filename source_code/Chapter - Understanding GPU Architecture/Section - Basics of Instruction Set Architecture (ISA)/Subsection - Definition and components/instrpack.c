#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#define OPCODE_BITS 8U      // supports 256 opcodes
#define REG_BITS    6U      // supports 64 registers per operand
#define NUM_OPDS    3U      // three-operand R-type
#define IMM_BITS    8U      // 8-bit immediate
#define PRED_BITS   2U      // two-bit predicate mask
#define META_BITS   (32U - (OPCODE_BITS + NUM_OPDS*REG_BITS + IMM_BITS + PRED_BITS))

// Pack fields into a 32-bit instruction. Returns true on success.
bool pack_instruction(uint8_t opcode, uint8_t regs[NUM_OPDS],
                      uint8_t imm, uint8_t pred, uint32_t *instr_out) {
    // range checks
    if (opcode >= (1U << OPCODE_BITS)) return false;
    for (unsigned i = 0; i < NUM_OPDS; i++) {
        if (regs[i] >= (1U << REG_BITS)) return false;
    }
    if (imm >= (1U << IMM_BITS)) return false;
    if (pred >= (1U << PRED_BITS)) return false;

    uint32_t w = 0;
    unsigned shift = 0;

    // pack opcode
    w |= ((uint32_t)opcode & ((1U << OPCODE_BITS) - 1U)) << shift;
    shift += OPCODE_BITS;

    // pack registers
    for (unsigned i = 0; i < NUM_OPDS; i++) {
        w |= ((uint32_t)regs[i] & ((1U << REG_BITS) - 1U)) << shift;
        shift += REG_BITS;
    }

    // pack immediate
    w |= ((uint32_t)imm & ((1U << IMM_BITS) - 1U)) << shift;
    shift += IMM_BITS;

    // pack predicate
    w |= ((uint32_t)pred & ((1U << PRED_BITS) - 1U)) << shift;
    shift += PRED_BITS;

    // pack metadata (reserved)
    w |= 0U << shift; // reserved bits

    *instr_out = w;
    return true;
}