import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

# models
from style_encoder import StyleEncoder, initialize_weights
from content_encoder import ContentEncoder
from discriminator import Discriminator
from decoder import DummyAutoregressiveDecoder
from losses import infoNCE_loss, margin_loss, adversarial_loss, disentanglement_loss
from dummy_dataloader import get_dummy_dataloader

# hyperparameters
EPOCHS = 50
BATCH_SIZE = 8
LR_GEN = 1e-4  # Learning rate per Encoders + Decoder
LR_DISC = 1e-4  # Learning rate per Discriminator
TRANSFORMER_DIM = 256
NUM_FRAMES = 4 # S
STFT_T, STFT_F = 287, 513 # Dimensioni STFT (assumendo n_fft=1024)
CQT_T, CQT_F = 287, 84   # Dimensioni CQT

# loss weights
LAMBDA_RECON = 10.0
LAMBDA_INFO_NCE = 1.0
LAMBDA_MARGIN = 1.0
LAMBDA_DISENTANGLE = 0.1
LAMBDA_ADV_GEN = 0.5 # Peso per la loss avversaria del generatore

MODEL_SAVE_PATH = "./saved_models"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load models
    style_encoder = StyleEncoder(transformer_dim=TRANSFORMER_DIM).to(device)
    content_encoder = ContentEncoder(transformer_dim=TRANSFORMER_DIM).to(device)
    decoder = DummyAutoregressiveDecoder(
        content_dim=TRANSFORMER_DIM,
        style_dim=TRANSFORMER_DIM,
        output_dims=(2, STFT_T, STFT_F + CQT_F) # Output combinato STFT+CQT
    ).to(device)
    discriminator = Discriminator(input_dim=TRANSFORMER_DIM).to(device)

    # initialize weights
    initialize_weights(style_encoder)
    initialize_weights(content_encoder)
    initialize_weights(decoder)
    initialize_weights(discriminator)

    # optimizer for generators (style encoder, content encoder, decoder)
    optimizer_G = optim.Adam(
        list(style_encoder.parameters()) + list(content_encoder.parameters()) + list(decoder.parameters()),
        lr=LR_GEN, betas=(0.5, 0.999)
    )
    
    # optimizer for discriminator
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR_DISC, betas=(0.5, 0.999))

    dataloader = get_dummy_dataloader(batch_size=BATCH_SIZE, num_frames=NUM_FRAMES, T=STFT_T, F=STFT_F + CQT_F)
    
    # recon loss
    recon_loss_fn = nn.L1Loss()
    
    best_loss = float('inf')

    # train loop
    for epoch in range(EPOCHS):
        
        style_encoder.train()
        content_encoder.train()
        decoder.train()
        discriminator.train()
        
        for i, (x, labels) in enumerate(tqdm(dataloader, unit="batch", desc=f"Epoch {epoch+1}/{EPOCHS}")):
            x, labels = x.to(device), labels.to(device) # x: (B, S, 2, T, F)

            # ================================================================== #
            #                             Discriminator                          #
            # ================================================================== #
            optimizer_D.zero_grad()
            
            # compute embeddings with no_grad to avoid backpropagation through the encoders
            with torch.no_grad():
                style_emb, class_emb = style_encoder(x, labels)
                content_emb = content_encoder(x)
            
            # adversarial loss for the discriminator
            discriminator_loss, _ = adversarial_loss(style_emb.detach(), class_emb.detach(), 
                                                     content_emb.detach(), discriminator, labels, 
                                                     compute_for_discriminator=True)
            
            discriminator_loss.backward()
            optimizer_D.step()


            # ================================================================== #
            #               Generators (Style Encoder, Content Encoder)          #
            # ================================================================== #
            optimizer_G.zero_grad()

            # forward pass
            style_emb, class_emb = style_encoder(x, labels)
            content_emb = content_encoder(x)

            # adversarial loss for the generator
            _, adv_generator_loss = adversarial_loss(style_emb, class_emb, content_emb, discriminator, labels,
                                                 compute_for_discriminator=False)

            # disentanglement loss
            disent_loss = disentanglement_loss(style_emb, content_emb.mean(dim=1), use_hsic=True)

            if len(torch.unique(labels)) > 1:
                # contrastive losses
                loss_infonce = infoNCE_loss(style_emb, labels)
                loss_margin = margin_loss(class_emb)
            else:
                raise ValueError("Labels must contain at least two unique classes for contrastive losses.")

            # --- Loss di Ricostruzione (Decoder) ---
            # Il decoder deve ricostruire lo spettrogramma originale
            reconstructed_spec = decoder(content_emb, style_emb, target_stft=x)
            loss_recon = recon_loss_fn(reconstructed_spec, x)
            
            # total generator loss
            total_gen_loss = (
                LAMBDA_RECON * loss_recon +
                LAMBDA_INFO_NCE * loss_infonce +
                LAMBDA_MARGIN * loss_margin +
                LAMBDA_DISENTANGLE * disent_loss +
                LAMBDA_ADV_GEN * adv_generator_loss
            )

            total_gen_loss.backward()
            optimizer_G.step()

            # loss logging
            if i % 1 == 0:
                tqdm.write(
                    f"Batch {i}/{len(dataloader)} | "
                    f"Discriminator loss: {discriminator_loss.item():.4f} | "
                    f"Total Generator loss: {total_gen_loss.item():.4f} | "
                    f"Reconstruction loss: {loss_recon.item():.4f} | "
                    f"Adversary Generator loss: {adv_generator_loss.item():.4f} | "
                    f"Disentanglement loss: {disent_loss.item():.4f}"
                )
                
        # saving best model
        current_recon_loss = loss_recon.item()
        if current_recon_loss < best_loss:
            best_loss = current_recon_loss
            print(f"\nNew best reconstruction loss: {best_loss:.4f}. Saving model...")
            
            torch.save({
                'epoch': epoch,
                'style_encoder_state_dict': style_encoder.state_dict(),
                'content_encoder_state_dict': content_encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'best_loss': best_loss,
            }, os.path.join(MODEL_SAVE_PATH, 'best_model.pth'))