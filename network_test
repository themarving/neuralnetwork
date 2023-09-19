import org.junit.Test;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.util.Random;
import neuralnetwork.*;

public class NetworkTest {

    @Test
    public void testing() {
        Random random = new Random();

        // test data 6x12 matrix
        Double[][] X = {
            {-41.7, 8.2, -2.3, 0.1, 7.5, -9.9, 6.3, -555.4, 3.2, 2.8, -1.9, -62.7},
            {-0.5, 921.0, -3.6, 40.1, -8.8, -0.8, 1.3, -2.1, 9.9, 5.4, -4.2, -6.3},
            {-8.5, 3.7, -5.7, 1.8, -8.4, -7.9, 3.1, 2.6, 60.1, -6.2, -7.3, -1.1},
            {-53.9, -35.9, 7.3, 4.4, -9.3, 0.2, 1.5, 6.9, -1.7, 96.6, -92.7, -6.5},
            {-1.5, -9.0, -7.1, 0.8, 5.6, -8.1, -7.7, -23.5, 6.5, -8.3, 5.2, -3.3},
            {7.9, 9.1, 8.8, -5.1, 6.4, 100.5, 5.1, -23.2, -0.6, 3.6, 4.9, 9.5}
        };


        /* TESTING PERCEPTRON NETWORK */
        TestNetwork testNetwork = new TestNetwork();

        assertDoesNotThrow(() -> testNetwork.activateNetwork(X));


        /* TESTING ACTIVATION FUNCTIONS */
        Double[][] output = testNetwork.activateNetwork(X);
        Double[][] rectifiedLU = ActivationFunctions.rectifiedLU(output);
        Double[][] softMax = ActivationFunctions.softMax(output);

        assertTrue(verifyRectifiedLU(rectifiedLU));
        assertTrue(verifySoftMax(softMax));
        assertFalse(verifySoftMax(rectifiedLU));


        /* TESTING LOSS FUNCTION WITH CLASS TARGETS */
        Integer[] classTargets = new Integer[output.length];

        // creating random class targets for loss calculation
        for (int i = 0; i < output.length; i++) {
            classTargets[i] = random.nextInt(output[0].length - 1);
        }

        Double loss = new LossFunction().calculateLoss(output, classTargets);

        assertNotNull(loss);
    }


    private class TestNetwork {

        private NeuronLayer<Double> inputLayer = null;
        private NeuronLayer<Double> hiddenLayer1 = null;
        private NeuronLayer<Double> hiddenLayer2 = null;
        private NeuronLayer<Double> hiddenLayer3 = null;
        private NeuronLayer<Double> outputLayer = null;

        public Double[][] activateNetwork(Double[][] X) {
            Double[][] output = null;

            try {
                // forwarding the perceptron network
                inputLayer = new NeuronLayer<>(X, 12, 5000);
                hiddenLayer1 = new NeuronLayer<>(inputLayer.forward(), 5000, 350);
                hiddenLayer2 = new NeuronLayer<>(hiddenLayer1.forward(), 350, 80);
                hiddenLayer3 = new NeuronLayer<>(hiddenLayer2.forward(), 80, 45);
                outputLayer = new NeuronLayer<>(hiddenLayer3.forward(), 45, 3);

                output = outputLayer.forward();
            } catch (InvalidDataException e) {

            }

            return output;
        }
    }

    private boolean verifyRectifiedLU(Double[][] data) {
        for (Double[] row : data) {
            for (int i = 0; i < row.length; i++) {
                if (row[i] < 0.0) {
                    return false;
                }
            }
        }
        return true;
    }

    private boolean verifySoftMax(Double[][] data) {
        for (Double[] row : data) {
            for (int i = 0; i < row.length; i++) {
                if (row[i] < 0.0 || row[i] > 1.0) {
                    return false;
                }
            }
        }
        return true;
    }

}
