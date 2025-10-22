module issue_dispatch_arb #(
  parameter W = 32,           // number of warps tracked
  parameter U = 6,            // number of functional unit types
  parameter WID = $clog2(W)   // bits to encode warp id
)(
  input  wire                 clk,
  input  wire                 rst,
  input  wire [W-1:0]         ready_vec,      // warp-level ready bit
  input  wire [W-1:0]         scoreboard,     // 1 => operands ready
  input  wire [U-1:0]         unit_avail,     // unit free vector
  input  wire [U-1:0]         target_vec [W-1:0], // per-warp 1-hot target
  output reg  [WID-1:0]       disp_warp_id,   // warp selected to dispatch
  output reg  [U-1:0]         disp_unit,      // unit selected for dispatch (1-hot)
  output reg                  disp_valid      // handshake to downstream
);

  // internal rotated priority pointer for fairness
  reg [W-1:0] rr_mask; // one-hot rotated; 1 means highest priority
  integer i;

  // compute eligibility per warp combinationally
  wire [W-1:0] eligible;
  genvar g;
  generate
    for (g = 0; g < W; g = g + 1) begin : ELIG
      assign eligible[g] = ready_vec[g] & scoreboard[g];
    end
  endgenerate

  // one-hot round-robin selection
  reg [W-1:0] sel_vec;
  always @(*) begin
    sel_vec = 0;
    for (i = 0; i < W; i = i + 1) begin
      int idx = (i + $clog2(rr_mask)) % W;
      if (eligible[idx]) begin
        sel_vec[idx] = 1'b1;
        disable for;
      end
    end
  end

  // decode selected warp and issue unit
  always @(*) begin
    disp_valid = |sel_vec;
    disp_warp_id = 0;
    disp_unit = 0;
    if (disp_valid) begin
      for (i = 0; i < W; i = i + 1)
        if (sel_vec[i]) begin
          disp_warp_id = i[WID-1:0];
          disp_unit = target_vec[i] & unit_avail;
        end
    end
  end

  // rotate priority mask after successful dispatch
  always @(posedge clk or posedge rst) begin
    if (rst)
      rr_mask <= 1;
    else if (disp_valid)
      rr_mask <= {rr_mask[W-2:0], rr_mask[W-1]};
  end

endmodule