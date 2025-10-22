module warp_scheduler #(
  parameter WARP_CNT   = 64,            // number of resident warps
  parameter ISSUE_WIDTH = 2,            // number of warps to issue per cycle
  parameter IDX_W      = $clog2(WARP_CNT)
)(
  input  wire                clk,
  input  wire                rst_n,
  input  wire [WARP_CNT-1:0] ready_mask,   // candidate ready warps
  input  wire [WARP_CNT-1:0] scoreboard,   // 1 => blocked (can't issue)
  output reg  [WARP_CNT-1:0] sel_mask,     // one-hot/multi-hot selected warps
  output reg  [IDX_W-1:0]    sel_idx,      // encoded index of first selection
  output reg                 valid         // true if at least one issued
);

  reg [IDX_W-1:0] rr_ptr;                  // round-robin pointer
  wire [WARP_CNT-1:0] eligible = ready_mask & ~scoreboard;

  // --- rotate right helper ---
  function [WARP_CNT-1:0] rotate_right;
    input [WARP_CNT-1:0] vec;
    input [IDX_W-1:0] rot;
    begin
      rotate_right = (vec >> rot) | (vec << (WARP_CNT - rot));
    end
  endfunction

  // --- rotate left helper ---
  function [WARP_CNT-1:0] rotate_left;
    input [WARP_CNT-1:0] vec;
    input [IDX_W-1:0] rot;
    begin
      rotate_left = (vec << rot) | (vec >> (WARP_CNT - rot));
    end
  endfunction

  reg [WARP_CNT-1:0] rotated;
  reg [WARP_CNT-1:0] pick;
  integer i, count;

  always @(*) begin
    rotated = rotate_right(eligible, rr_ptr);
    pick = 0;
    count = 0;
    for (i = 0; i < WARP_CNT && count < ISSUE_WIDTH; i = i + 1) begin
      if (rotated[i]) begin
        pick[i] = 1'b1;
        count = count + 1;
      end
    end
    sel_mask = rotate_left(pick, rr_ptr);
    valid = (count != 0);
  end

  // encode first selected warp
  always @(*) begin
    sel_idx = 0;
    for (i = 0; i < WARP_CNT; i = i + 1)
      if (sel_mask[i]) begin
        sel_idx = i[IDX_W-1:0];
        disable for;
      end
  end

  // round-robin pointer update
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      rr_ptr <= 0;
    else if (valid)
      rr_ptr <= rr_ptr + ISSUE_WIDTH[IDX_W-1:0];
  end

endmodule