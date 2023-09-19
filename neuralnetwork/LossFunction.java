package neuralnetwork;

public class LossFunction {
    
    /* calculating the loss that would be used to train the network (i.e. comparing
       loss with actual expected output (entropy -> measure of "wrongness")) */
    public Double calculateLoss(Double[][] input, Integer[] classTargets) {
        // checking that our class target array is complete
        if (input.length != classTargets.length) {
            return null;
        }

        Double[] lossValues = new Double[input.length];

        // calculating loss for each target
        for (int i = 0; i < input.length; i++) {
            // guarding agains log of 0 (infinity)
            if (input[i][classTargets[i]] == 0.0) {
                lossValues[i] = 0.0;
            }
            else {
                lossValues[i] = -(Math.log(input[i][classTargets[i]]));
            }
        }

        Double lossSum = 0.0;

        for (Double num : lossValues) {
            lossSum += num;
        }

        return lossSum / lossValues.length;
    }

}
