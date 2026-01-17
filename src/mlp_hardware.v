module neuron_int8 (
    input clk,                  // Clock signal
    input rst,                  // Reset signal
    input signed [7:0] weight,  // 8-bit Integer Weight
    input signed [7:0] input_val, // 8-bit Integer Input
    output reg signed [19:0] accumulated_sum // Output Sum
);

    reg signed [15:0] product;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            accumulated_sum <= 0;
            product <= 0;
        end else begin
            // 1. Multiply (Int8 * Int8)
            product <= weight * input_val;
            
            // 2. Accumulate (Add to running total)
            accumulated_sum <= accumulated_sum + product;
        end
    end
endmodule