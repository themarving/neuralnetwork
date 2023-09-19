package neuralnetwork;

import java.util.Arrays;

public class ActivationFunctions {

    // activation function utilizing a rectified linear unit (ReLU)
    public static Double[][] rectifiedLU(Double[][] input) {
        Double[][] output = new Double[input.length][input[0].length];

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                // cutting of values below zero
                output[i][j] = Math.max(0.0, input[i][j]);
            }
        }

        return output;
    } 

    /* activation function utilizing softmax activation,
       thus we normalize the output to probabilities between 0 and 1 */
    public static Double[][] softMax(Double[][] input) {
        Double[][] output = new Double[input.length][input[0].length];
        Double[] rowMaxValue = new Double[input.length];

        // finding the maximum value in each input array
        for (int i = 0; i < input.length; i++) {
            // using streams to find the maximum value for each array
            rowMaxValue[i] = Arrays.stream(input[i])
                    .max((a, b) -> a.compareTo(b))
                    // guarding against type mismatch (optional/double)
                    .orElse(Double.MIN_VALUE);
        }

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                /* exponentiating all input values with euler's number
                   while subtracting max number from each element (guarding against overflow) */
                output[i][j] = (input[i][j] - rowMaxValue[i]) * Math.E;
            }
        }

        Double[] normalizationBase = new Double[input.length];

        // summing each exponentiated array
        for (int i = 0; i < output.length; i++) {
            normalizationBase[i] = 0.0;

            for (int j = 0; j < output[i].length; j++) {
                normalizationBase[i] += output[i][j];
            }
        }

        // normalizing exponentiated values
        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[0].length; j++) {
                /* Math.abs not strictly neccessary, but makes output
                   look better (-0.0 -> 0.0) */
                output[i][j] = Math.abs(output[i][j] / normalizationBase[i]);
            }
        }

        return output;
    } 

}
