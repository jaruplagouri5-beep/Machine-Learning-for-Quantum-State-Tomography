`timescale 1ns / 1ps

module testbench;

    // Inputs
    reg clk;
    reg rst;
    reg signed [7:0] weight;
    reg signed [7:0] input_val;

    // Outputs
    wire signed [19:0] output_sum;

    // Connect to the Neuron Module
    neuron_int8 uut (
        .clk(clk),
        .rst(rst),
        .weight(weight),
        .input_val(input_val),
        .accumulated_sum(output_sum)
    );

    // Generate Clock (Switch 0->1 every 5ns)
    always #5 clk = ~clk;

    initial begin
        // --- THIS GENERATES THE .VCD FILE ---
        $dumpfile("outputs/simulation.vcd"); // Where to save
        $dumpvars(0, testbench);             // What to record

        // Initialize
        clk = 0;
        rst = 1;
        weight = 0;
        input_val = 0;

        // Reset for 10ns
        #10;
        rst = 0;

        // Test Case 1: 10 * 2 = 20
        weight = 10;
        input_val = 2;
        #10; 

        // Test Case 2: -5 * 3 = -15 (Accumulated: 20 - 15 = 5)
        weight = -5;
        input_val = 3;
        #10;

        // Test Case 3: 20 * 4 = 80 (Accumulated: 5 + 80 = 85)
        weight = 20;
        input_val = 4;
        #10;

        // Finish Simulation
        $finish;
    end
endmodule