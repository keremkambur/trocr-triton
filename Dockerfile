FROM nvcr.io/nvidia/tritonserver:24.10-py3

# Install required packages
RUN pip install torch transformers Pillow

# Set environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Expose Triton's ports
EXPOSE 8000
EXPOSE 8001
EXPOSE 8002

# Start Triton
CMD ["tritonserver", "--model-repository=/models"]
