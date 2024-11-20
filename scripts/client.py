import tritonclient.http as httpclient
import numpy as np
import sys

def main():
    try:
        # Create client
        client = httpclient.InferenceServerClient(
            url="localhost:8010",
            verbose=True
        )

        # Check server status
        if not client.is_server_ready():
            print("Server is not ready!")
            sys.exit(1)
        
        if not client.is_model_ready("trocr"):
            print("Model is not ready!")
            sys.exit(1)

        # Prepare filename
        filename = "form3.png"  # Just the filename, not the full path

        # # Create input object
        # input_data = httpclient.InferInput("filename", [1], "TEXT")
        # input_data.set_data_from_numpy(np.array([filename], dtype=object))  # Pass plain string, not bytes
        

        np_input_data = np.asarray([filename], dtype=object)

        text = httpclient.InferInput('filename', [1], "BYTES")
        text.set_data_from_numpy(np_input_data.reshape([1]))
        # Create output object
        output = httpclient.InferRequestedOutput("text_output")

        # Send request
        response = client.infer("trocr", [text], outputs=[output])  # Pass input_data, not filename

        # Get result
        result = response.as_numpy("text_output")
        print(f"Recognized text: {result[0]}")  # Already string; no need to decode

    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()