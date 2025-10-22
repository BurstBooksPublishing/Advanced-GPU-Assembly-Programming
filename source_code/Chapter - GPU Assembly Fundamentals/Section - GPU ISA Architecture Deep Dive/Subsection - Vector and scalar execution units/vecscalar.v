module vec_scalar_unit #(
  parameter LANES = 32,                 // warp width
  parameter LANEW = 32                  // lane bitwidth
)(
  input  wire                   clk,
  input  wire                   rst_n,
  input  wire [LANES*LANEW-1:0] vec_a,  // flat vector input
  input  wire [LANES*LANEW-1:0] vec_b,
  input  wire [LANES-1:0]       vec_mask,// per-lane active mask
  input  wire                   sc_req,  // scalar op request
  input  wire [31:0]            sc_imm,  // scalar immediate
  output reg  [LANES*LANEW-1:0] vec_out, // vector result
  output reg  [LANES-1:0]       sc_flag  // scalar flags per-lane
);

  // lane registers and pipeline stage
  genvar i;
  generate
    for (i = 0; i < LANES; i = i + 1) begin : lane_blk
      reg [LANEW-1:0] a_reg, b_reg, sum_reg;
      reg             active;

      always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          a_reg   <= {LANEW{1'b0}};
          b_reg   <= {LANEW{1'b0}};
          sum_reg <= {LANEW{1'b0}};
          active  <= 1'b0;
          sc_flag[i] <= 1'b0;
        end else begin
          active <= vec_mask[i];
          if (active) begin
            a_reg <= vec_a[i*LANEW +: LANEW];
            b_reg <= vec_b[i*LANEW +: LANEW];
            sum_reg <= a_reg + b_reg; // pipeline stage
            sc_flag[i] <= (sc_req && (sum_reg[LANEW-1:LANEW-4] == sc_imm[3:0]));
          end
        end
      end

      always @(*) begin
        vec_out[i*LANEW +: LANEW] = active ? sum_reg : {LANEW{1'b0}};
      end
    end
  endgenerate

endmodule