import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="MNIST Digit Generator",
    page_icon="🔢",
    layout="centered"
)

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # Encoder: Input is [image + one-hot label]
        self.fc1 = nn.Linear(input_dim + num_classes, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # mu
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # logvar

        # Decoder: Input is [z + one-hot label]
        self.fc3 = nn.Linear(latent_dim + num_classes, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, y):
        # x: [batch, 784], y: [batch, num_classes]
        x = torch.cat([x, y], dim=1)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        # z: [batch, latent_dim], y: [batch, num_classes]
        z = torch.cat([z, y], dim=1)
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x.view(-1, 784), y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

@st.cache_resource
def load_model():
    """Load the trained VAE model"""
    try:
        checkpoint = torch.load('mnist_cvae_model.pth', map_location='cpu')
        model = VAE(latent_dim=checkpoint['latent_dim'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, checkpoint
    except FileNotFoundError:
        st.error("❌ Model file 'mnist_vae_model.pth' not found!")
        st.info("Please make sure the trained model file is in the same directory.")
        return None, None

def generate_digit_samples(model, num_samples=5, target_digit=0):
    """Generate digit-conditioned samples from the CVAE"""
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim)
        
        # One-hot encode the target digit
        y = torch.zeros(num_samples, 10)
        y[:, target_digit] = 1.0
        
        generated = model.decode(z, y)
        images = generated.view(-1, 28, 28).numpy()
        return images

def create_image_grid(images, title):
    """Create a grid of images"""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Sample {i+1}', fontsize=12)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def main():
    # Header
    st.title("🔢 MNIST Digit Generator")
    st.markdown("Generate handwritten digit images using a trained Variational Autoencoder")
    
    # Load model
    model, checkpoint = load_model()
    
    if model is None:
        st.stop()
    
    # Success message
    st.success("✅ Model loaded successfully!")
    
    # Main interface
    st.markdown("---")
    st.header("Generate Digit Images")
    
    # Create two columns for better layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input")
        
        # Number input
        selected_digit = st.selectbox(
            "Choose a digit to generate:",
            options=list(range(10)),
            index=0,
            help="Select a digit from 0 to 9"
        )
        
        # Display selected digit
        st.metric("Selected Digit", selected_digit)
        
        # Generate button
        generate_button = st.button(
            "🎨 Generate 5 Images", 
            type="primary",
            use_container_width=True
        )
        
        # Info about the model
        with st.expander("ℹ️ Model Info"):
            st.write(f"**Latent Dimensions:** {checkpoint['latent_dim']}")
            st.write(f"**Model Parameters:** {sum(p.numel() for p in model.parameters()):,}")
            st.write("**Note:** This is an unconditional VAE, so it generates random digit-like images rather than specific digits.")
    
    with col2:
        st.subheader("Generated Images")
        
        if generate_button:
            with st.spinner(f"Generating 5 images of digit {selected_digit}..."):
                # Generate images
                generated_images = generate_digit_samples(model, num_samples=5, target_digit=selected_digit)
                
                # Create and display the image grid
                fig = create_image_grid(
                    generated_images, 
                    f'Generated Images for Digit "{selected_digit}"'
                )
                st.pyplot(fig)
                
                # Store in session state for download
                st.session_state.last_generated = generated_images
                st.session_state.last_digit = selected_digit
                
                st.success(f"✨ Successfully generated 5 images for digit {selected_digit}!")
        
        # Show previous results if they exist
        elif 'last_generated' in st.session_state:
            st.info(f"Showing previous results for digit {st.session_state.last_digit}")
            fig = create_image_grid(
                st.session_state.last_generated, 
                f'Generated Images for Digit "{st.session_state.last_digit}"'
            )
            st.pyplot(fig)
        
        else:
            st.info("👆 Select a digit and click 'Generate 5 Images' to see results!")
    
    # Additional features
    st.markdown("---")
    
    # Download section
    if 'last_generated' in st.session_state:
        st.subheader("💾 Download Images")
        
        col1, col2, col3 = st.columns(3)
        
        with col2:
            if st.button("📥 Download All Images", use_container_width=True):
                # Create a zip file with all images
                import zipfile
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    for i, img in enumerate(st.session_state.last_generated):
                        # Convert to PIL Image and save
                        pil_img = Image.fromarray((img * 255).astype(np.uint8))
                        img_buffer = io.BytesIO()
                        pil_img.save(img_buffer, format='PNG')
                        zip_file.writestr(f'digit_{st.session_state.last_digit}_sample_{i+1}.png', img_buffer.getvalue())
                
                st.download_button(
                    label="📁 Download ZIP",
                    data=zip_buffer.getvalue(),
                    file_name=f"digit_{st.session_state.last_digit}_samples.zip",
                    mime="application/zip"
                )
    
    # Instructions
    st.markdown("---")
    st.subheader("📋 How to Use")
    
    instructions = """
    1. **Select a digit** from the dropdown (0-9)
    2. **Click 'Generate 5 Images'** to create samples
    3. **View the results** in the image grid
    4. **Download** the images if needed
    
    **Note:** Since this is an unconditional VAE, it generates random handwritten digit-like images 
    rather than specific digits. The model learned the general structure of handwritten digits 
    from the MNIST dataset.
    """
    
    st.markdown(instructions)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with PyTorch VAE • Deployed with Streamlit • Trained on MNIST Dataset"
        "</div>", 
        unsafe_allow_html=True
    )

main()
