module warp_arbiter #(
  parameter W = 32  // number of warps (power of two simplifies rotate)
)(
  input  wire                   clk,
  input  wire                   rst_n,
  input  wire [W-1:0]           ready_mask,   // 1 = warp ready after scoreboard
  input  wire [W-1:0]           fu_avail_mask,// functional-unit availability mask
  output reg  [W-1:0]           grant_mask,   // one-hot grant
  output reg  [$clog2(W)-1:0]   grant_idx     // encoded grant
);

  reg [$clog2(W)-1:0] pointer;               // rotate pointer for fairness

  // rotate ready candidates by pointer, mask by FU availability
  wire [W-1:0] candidates = (ready_mask & fu_avail_mask);
  wire [W-1:0] rotated = (candidates << pointer) | (candidates >> (W - pointer));

  // priority encoder for rotated vector
  integer i;
  reg [W-1:0] one_hot_rot;
  always @(*) begin
    one_hot_rot = {W{1'b0}};
    for (i = 0; i < W; i = i + 1) begin
      if (rotated[i]) begin one_hot_rot[i] = 1'b1; break; end
    end
  end

  // unrotate one-hot to original index
  wire [W-1:0] one_hot = (one_hot_rot >> pointer) | (one_hot_rot << (W - pointer));

  // Encode one-hot to index
  integer j;
  reg [$clog2(W)-1:0] idx_enc;
  always @(*) begin
    idx_enc = {($clog2(W)){1'b0}};
    for (j = 0; j < W; j = j + 1) begin
      if (one_hot[j]) idx_enc = j[$clog2(W)-1:0];
    end
  end

  // update pointer and outputs
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      pointer  <= 0;
      grant_mask <= {W{1'b0}};
      grant_idx  <= 0;
    end else begin
      grant_mask <= one_hot;
      grant_idx  <= idx_enc;
      // advance pointer to next after granted for fairness
      if (one_hot != {W{1'b0}})
        pointer <= idx_enc + 1;
    end
  end

endmodule