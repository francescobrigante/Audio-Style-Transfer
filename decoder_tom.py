import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

class LightweightDecoder(nn.Module):
    def __init__(self, d=256, d_hidden=128, num_channels=16, S=4):
        super().__init__()
        self.S = S
        self.conv_in = nn.Conv2d(
            in_channels=2, out_channels=num_channels, kernel_size=3, stride=2, padding=1
        )
        self.ln1 = nn.LayerNorm([num_channels, 144, 257])
        self.linear_in = nn.Linear(num_channels * 144 * 257, d)

        self.attention = nn.MultiheadAttention(embed_dim=d, num_heads=2, batch_first=True)
        self.gru = nn.GRU(input_size=d, hidden_size=d_hidden, num_layers=1, batch_first=True)

        self.linear_out = nn.Linear(d_hidden, num_channels * 144 * 257)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                num_channels, 8, kernel_size=3, stride=2, padding=1, output_padding=0
            ),  # [B, 8, 287, 513]
            nn.ReLU(),
            nn.Conv2d(8, 2, kernel_size=1, stride=1, padding=0)  # [B, 2, 287, 513]
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, y_prev, class_emb, content_emb, hidden=None):
        batch_size = content_emb.size(0)
        outputs = []

        with autocast('cuda'):
            for s in range(self.S):
                x = F.relu(self.ln1(self.conv_in(y_prev[:, s, :, :, :])))  # [B, num_channels, 144, 257]
                x = x.view(x.size(0), -1)  # [B, num_channels*144*257]
                x = self.dropout(self.linear_in(x))  # [B, d]

                context = torch.stack(
                    [content_emb[:, s, :], class_emb.expand(batch_size, -1)], dim=1
                )  # [B, 2, d]
                attn_output, _ = self.attention(
                    x.unsqueeze(1), context, context
                )  # [B, 1, d]
                attn_output = self.dropout(attn_output.squeeze(1))  # [B, d]

                gru_output, hidden = self.gru(attn_output.unsqueeze(1), hidden)  # [B, 1, d_hidden]
                gru_output = gru_output.squeeze(1)  # [B, d_hidden]

                out = self.linear_out(gru_output)  # [B, num_channels*144*257]
                out = out.view(-1, num_channels, 144, 257)  # [B, num_channels, 144, 257]
                out = self.deconv(out)  # [B, 2, 287, 513]

                outputs.append(out.unsqueeze(2))  # [B, 2, 1, 287, 513]
                y_prev[:, s, :, :, :] = out.detach()  # Aggiorna y_prev

            output = torch.cat(outputs, dim=2)  # [B, 2, S, 287, 513]

            if (y_prev.shape[0] == output.shape[0] and
                y_prev.shape[1] == output.shape[2] and
                y_prev.shape[2] == output.shape[1] and
                y_prev.shape[3] == output.shape[3] and
                y_prev.shape[4] == output.shape[4]):
                print("Dimensioni corrette")
            else:
                print(f"Problema dimensioni: input {y_prev.shape}, output {output.shape}")

        return output, hidden

def compute_loss(output, target, S=4):
    mse_loss = sum(F.mse_loss(output[:, :, s, :, :], target[:, :, s, :, :]) for s in range(S))
    mag_output = torch.sqrt(output[:, 0, :, :, :]**2 + output[:, 1, :, :, :]**2)
    mag_target = torch.sqrt(target[:, 0, :, :, :]**2 + target[:, 1, :, :, :]**2)
    mag_loss = sum(F.mse_loss(mag_output[:, s, :, :], mag_target[:, s, :, :]) for s in range(S))
    sc_loss = sum(
        torch.norm(mag_output[:, s, :, :] - mag_target[:, s, :, :], p='fro') /
        torch.norm(mag_target[:, s, :, :], p='fro') for s in range(S)
    )
    total_loss = 0.45 * mse_loss + 0.45 * mag_loss + 0.1 * sc_loss
    return total_loss

if __name__ == "__main__":
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    batch_size = 4
    d = 256
    d_hidden = 128
    num_channels = 16
    S = 4

    decoder = LightweightDecoder(d=d, d_hidden=d_hidden, num_channels=num_channels, S=S).cuda()
    y_prev = torch.randn(batch_size, S, 2, 287, 513).cuda()
    class_emb = torch.randn(d).cuda()
    content_emb = torch.randn(batch_size, S, d).cuda()
    target = torch.randn(batch_size, 2, S, 287, 513).cuda()
    hidden = None

    output, hidden = decoder(y_prev, class_emb, content_emb, hidden)
    print(f"Output shape: {output.shape}")
    loss = compute_loss(output, target, S=4)
    print(f"Loss: {loss.item()}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")