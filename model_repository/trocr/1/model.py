import triton_python_backend_utils as pb_utils
import numpy as np
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import os

class TritonPythonModel:
    def initialize(self, args):
        """Initialize the model"""
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
        self.model.to('cuda')
        self.model.eval()
        
        # You might want to set this as an environment variable or config parameter
        self.image_dir = os.getenv('IMAGES_ROOT_DIR')  # Update this to your images directory

    
    def finalize(self):
        """Clean up"""
        pass
    
    def execute(self, requests):
        """Execute the model"""
        responses = []
        
        for request in requests:
            try:
                # Debug: Log incoming request
                input_tensor = pb_utils.get_input_tensor_by_name(request, "filename")
                print(f"Input tensor: {input_tensor}")
                print(f"Input data type: {input_tensor.as_numpy().dtype}")
                print(f"Input shape: {input_tensor.as_numpy().shape}")
                
                # Get filename from request
                filename = input_tensor.as_numpy()[0].decode('utf-8')
                print(f"Received filename: {filename}")
                
                # Construct full path
                image_path = os.path.join(self.image_dir, filename)
                
                # Load and process image
                image = Image.open(image_path).convert('RGB')
                
                # Process image using TrOCR processor
                pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to('cuda')
                
                # Generate text
                with torch.no_grad():
                    generated_ids = self.model.generate(pixel_values)
                    generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Create output tensor
                output_tensor = pb_utils.Tensor("text_output", np.array(generated_text, dtype=object))
            
            except Exception as e:
                # Handle errors
                error_message = f"Error processing image {filename}: {str(e)}"
                output_tensor = pb_utils.Tensor("text_output", np.array([error_message], dtype=object))
                print(e.__traceback__)
            
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)
        
        return responses