package neuralnetwork;

public class InvalidDataException extends Exception {
    String msg;
    
    public InvalidDataException(String msg) {
        System.out.println("ERROR: " + msg);
    }

    public String getMsg() {
        return msg;
    }

}
