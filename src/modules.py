import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class PatchEmbedding(nn.Module):
  """Embeds input images into patch representations with positional embeddings for transformer models."""
  def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
      super().__init__()
      self.patcher = nn.Sequential(
          # We use conv for doing the patching
          nn.Conv2d(
              in_channels=in_channels,
              out_channels=embed_dim,
              # if kernel_size = stride -> no overlap
              kernel_size=patch_size,
              stride=patch_size
          ),
          # Linear projection of Flattened Patches. We keep the batch and the channels (b,c,h,w)
          nn.Flatten(2))
      self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad=True)
      self.position_embeddings = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad=True)
      self.dropout = nn.Dropout(p=dropout)

  def forward(self, x):
      # Create a copy of the cls token for each of the elements of the BATCH
      cls_token = self.cls_token.expand(x.shape[0], -1, -1)
      # Create the patches
      x = self.patcher(x).permute(0, 2, 1)
      # Unify the position with the patches
      x = torch.cat([cls_token, x], dim=1)
      # Patch + Position Embedding
      x = self.position_embeddings + x
      x = self.dropout(x)
      return x
      

class MultiHeadSelfAttention(nn.Module):
    """MulitHeadSelfAttention architecture for the different modifications"""
    def __init__(self, emb_dim, num_heads, modification):
        super(MultiHeadSelfAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        assert emb_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."
        self.modification = modification
        self.attn_drop = nn.Dropout(0.1)

        if self.modification == "unchanged": 
            # Traditional Attention Mechanism.
            self.query = nn.Linear(emb_dim, emb_dim)
            self.key = nn.Linear(emb_dim, emb_dim)
            self.value = nn.Linear(emb_dim, emb_dim)
            self.out = nn.Linear(emb_dim, emb_dim)
        
        if self.modification == "Wqk": 
            # Query and key matrices are collapsed into a single matrix of the same size.
            self.qk = nn.Linear(emb_dim,emb_dim)
            self.value = nn.Linear(emb_dim, emb_dim)
            self.out = nn.Linear(emb_dim, emb_dim)

        if self.modification == "Wqk+noWvWo": 
            # In addition to the collapsed query and key matrices, value and projection matrices, are omitted without eliminating the substance of the attention mechanism.
            self.qk = nn.Linear(emb_dim,emb_dim)

        if self.modification == "symmetry" : 
            # The symmetric definition of a similarity matrix requires only half the parameters. This can be achieved by parameterizing a lower triangular matrix and multiplying it by its transpose.
            
            # Initialize one triangular matrix per attention head
            tril_size = (self.head_dim * (self.head_dim + 1)) // 2
            # Create parameters for all heads
            self.trainable_params = nn.Parameter(
                torch.randn(num_heads, tril_size)
            )
            self.value = nn.Linear(emb_dim, emb_dim)
            self.out = nn.Linear(emb_dim, emb_dim)
        
    def forward(self, x):
        bs, seq_len, emb_dim = x.size()


        if self.modification == "unchanged":
            Q = self.query(x).view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2) 
            K = self.key(x).view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)   
            V = self.value(x).view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2) 
         
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_drop(attn_weights)
    
            output = torch.matmul(attn_weights, V)  
            output = output.transpose(1, 2).contiguous().view(bs, seq_len, emb_dim)  
    
            
            output = self.out(output) 

        if self.modification == "Wqk":
            QK = self.qk(x).view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            QK = torch.matmul(QK, x.unsqueeze(1).transpose(2, 3))
            V = self.value(x).view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  

            attn_scores = QK / (self.head_dim ** 0.5)  
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_drop(attn_weights)
    
            output = torch.matmul(attn_weights, V)
            output = output.transpose(1, 2).contiguous().view(bs, seq_len, emb_dim)  
    
            output = self.out(output)  
            

        if self.modification == "Wqk+noWvWo":
            QK = self.qk(x).view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  
            QK = torch.matmul(QK, x.unsqueeze(1).transpose(2, 3))
            attn_scores = QK / (self.head_dim ** 0.5) 
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_drop(attn_weights)

            output = torch.matmul(attn_weights, x.unsqueeze(1))
            output = output.transpose(1, 2).contiguous().view(bs, seq_len, emb_dim)  
    

        if self.modification == "symmetry":
            V = self.value(x).view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            QK = torch.zeros(bs, self.num_heads, seq_len, seq_len, device=x.device)
            
            # Process each head separately
            for h in range(self.num_heads):
                # Create lower triangular matrix for this head
                tril = torch.zeros(self.head_dim, self.head_dim, device=x.device)
                indices = torch.tril_indices(self.head_dim, self.head_dim)
                tril[indices[0], indices[1]] = self.trainable_params[h]
                
                # Compute symmetric matrix using Cholesky decomposition
                Wqk = tril @ tril.T 
                
                # Project input for this head
                head_input = x.view(bs, seq_len, self.num_heads, self.head_dim)[:, :, h]
                
                # Compute attention scores for this head
                QK[:, h] = torch.matmul(
                    torch.matmul(head_input, Wqk),
                    head_input.transpose(-2, -1)
                )
            
            attn_scores = QK / (self.head_dim ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_drop(attn_weights)
            
            output = torch.matmul(attn_weights, V)
            output = output.transpose(1, 2).contiguous().view(bs, seq_len, emb_dim)
            
            output = self.out(output)  
            
        return output


class TransformerEncoderLayer(nn.Module):
    """Defines a one Transformer encoder layer"""
    def __init__(self, emb_dim, num_heads, mlp_dim, include_mlp,modification):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(emb_dim, num_heads,modification)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(0.1)
        self.include_mlp = include_mlp

        if include_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(emb_dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, emb_dim),
            )
       

    def forward(self, x):
        # Self-attention with residual connection
        attn_output = self.self_attn(x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # MLP with residual connection
        if self.include_mlp:
            mlp_output = self.mlp(x)
            x = x + self.dropout(mlp_output)
            x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    """Defines a multi-layer Transformer encoder."""
    def __init__(self, emb_dim, num_heads, num_layers, mlp_dim, include_mlp,modification):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(emb_dim, num_heads, mlp_dim, include_mlp,modification)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ReducedTransformer(nn.Module):
    """The modified transformer model for classification tasks."""
    def __init__(self, embed_dim, num_heads, num_layers, n_classes, patch_size, num_patches, dropout, in_channels, include_mlp, modification):
        super(ReducedTransformer, self).__init__()
        self.embedding = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)
        self.transformer_encoder = TransformerEncoder(embed_dim, num_heads, num_layers, embed_dim, include_mlp, modification)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:,0,:])
        return x


    