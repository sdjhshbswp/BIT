import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        self.pe = nn.Embedding(max_len, dim)
        nn.init.xavier_normal_(self.pe.weight.data)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class infoNCE(nn.Module):
    def __init__(self, temp_init, hdim):
        super().__init__()
        self.temp = nn.Parameter(torch.ones([]) * temp_init)

        self.weight_matrix = nn.Parameter(torch.randn((hdim, hdim)))
        nn.init.xavier_normal_(self.weight_matrix)

        self.tanh = nn.Tanh()
 
    def calculate_loss(self, query, item, neg_item):

        positive_logit = torch.sum((query @ self.weight_matrix) * item,
                                   dim=1,
                                   keepdim=True)
        negative_logits = (query @ self.weight_matrix) @ neg_item.transpose(
            -2, -1)

        positive_logit, negative_logits = self.tanh(positive_logit), self.tanh(
            negative_logits)

        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits),
                             dtype=torch.long,
                             device=query.device)

        return F.cross_entropy(logits / self.temp, labels, reduction='mean')

    def forward(self, query, click_item, neg_item, neg_query):
        query_loss = self.calculate_loss(query, click_item, neg_item)
        item_loss = self.calculate_loss(click_item, query, neg_query)

        return 0.5 * (query_loss + item_loss)


class feature_align(nn.Module):
    def __init__(self, temp_init, hdim):
        super().__init__()
        self.infoNCE_loss = infoNCE(temp_init, hdim)

    def filter_user_src_his(self, qry_his_emb, click_item_mask,
                            click_item_emb):
        qry_his_emb = qry_his_emb.unsqueeze(2).expand(-1, -1,
                                                      click_item_mask.size(2),
                                                      -1)

        src_his_query_emb = torch.masked_select(
            qry_his_emb.clone(),
            click_item_mask.unsqueeze(-1)).reshape(-1, qry_his_emb.size(-1))
        src_his_click_item_emb = torch.masked_select(click_item_emb.clone(), click_item_mask.unsqueeze(-1))\
            .reshape(-1, click_item_emb.size(-1))

        return src_his_query_emb, src_his_click_item_emb

    def forward(self, align_loss_input, query_emb, click_item_mask,
                q_click_item_emb):
        neg_item_emb, neg_query_emb = align_loss_input
        # neg_item_emb.shape: [neg_item_num, emb_dim]
        # neg_query_emb.shape: [neg_item_num, emb_dim]
        src_his_query_emb, src_his_click_item_emb = self.filter_user_src_his(
            query_emb, click_item_mask, q_click_item_emb)
        # src_his_query_emb.shape: [src_his_num * batch_size, emb_dim]
        # src_his_click_item_emb.shape: [src_his_num * batch_size, emb_dim]
        align_loss = self.infoNCE_loss(src_his_query_emb,
                                       src_his_click_item_emb, neg_item_emb,
                                       neg_query_emb)
        return align_loss


class FullyConnectedLayer(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_unit,
                 batch_norm=False,
                 activation='relu',
                 sigmoid=False,
                 dropout=None):
        super(FullyConnectedLayer, self).__init__()
        assert len(hidden_unit) >= 1
        self.sigmoid = sigmoid

        layers = []
        layers.append(nn.Linear(input_size, hidden_unit[0]))

        for i, h in enumerate(hidden_unit[:-1]):
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_unit[i]))

            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            elif activation.lower() == 'leakyrelu':
                layers.append(nn.LeakyReLU())
            else:
                raise NotImplementedError

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(hidden_unit[i], hidden_unit[i + 1]))

        self.fc = nn.Sequential(*layers)
        if self.sigmoid:
            self.output_layer = nn.Sigmoid()

    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.sigmoid else self.fc(x)


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 embed_dims,
                 dropout,
                 output_layer=True,
                 batch_norm=False):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class PLE_layer(nn.Module):
    def __init__(self, orig_input_dim, bottom_mlp_dims, tower_mlp_dims,
                 task_num, shared_expert_num, specific_expert_num,
                 dropout) -> None:
        super().__init__()
        self.embed_output_dim = orig_input_dim
        self.task_num = task_num
        self.shared_expert_num = shared_expert_num
        self.specific_expert_num = specific_expert_num
        self.layers_num = len(bottom_mlp_dims)

        self.task_experts = [[0] * self.task_num
                             for _ in range(self.layers_num)]
        self.task_gates = [[0] * self.task_num for _ in range(self.layers_num)]
        self.share_experts = [0] * self.layers_num
        self.share_gates = [0] * self.layers_num
        for i in range(self.layers_num):
            input_dim = self.embed_output_dim if 0 == i else bottom_mlp_dims[i
                                                                             -
                                                                             1]
            self.share_experts[i] = torch.nn.ModuleList([
                MultiLayerPerceptron(input_dim, [bottom_mlp_dims[i]],
                                     dropout,
                                     output_layer=False,
                                     batch_norm=False)
                for k in range(self.shared_expert_num)
            ])
            self.share_gates[i] = torch.nn.Sequential(
                torch.nn.Linear(
                    input_dim,
                    shared_expert_num + task_num * specific_expert_num),
                torch.nn.Softmax(dim=1))
            for j in range(task_num):
                self.task_experts[i][j] = torch.nn.ModuleList([
                    MultiLayerPerceptron(input_dim, [bottom_mlp_dims[i]],
                                         dropout,
                                         output_layer=False,
                                         batch_norm=False)
                    for k in range(self.specific_expert_num)
                ])
                self.task_gates[i][j] = torch.nn.Sequential(
                    torch.nn.Linear(input_dim,
                                    shared_expert_num + specific_expert_num),
                    torch.nn.Softmax(dim=1))
            self.task_experts[i] = torch.nn.ModuleList(self.task_experts[i])
            self.task_gates[i] = torch.nn.ModuleList(self.task_gates[i])

        self.task_experts = torch.nn.ModuleList(self.task_experts)
        self.task_gates = torch.nn.ModuleList(self.task_gates)
        self.share_experts = torch.nn.ModuleList(self.share_experts)
        self.share_gates = torch.nn.ModuleList(self.share_gates)

        self.tower = torch.nn.ModuleList([
            MultiLayerPerceptron(bottom_mlp_dims[-1],
                                 tower_mlp_dims,
                                 dropout,
                                 output_layer=False,
                                 batch_norm=False) for i in range(task_num)
        ])

    def forward(self, emb):
        task_fea = [emb for i in range(self.task_num + 1)]
        for i in range(self.layers_num):
            share_output = [
                expert(task_fea[-1]).unsqueeze(1)
                for expert in self.share_experts[i]
            ]
            task_output_list = []
            for j in range(self.task_num):
                task_output = [
                    expert(task_fea[j]).unsqueeze(1)
                    for expert in self.task_experts[i][j]
                ]
                task_output_list.extend(task_output)
                mix_ouput = torch.cat(task_output + share_output, dim=1)
                gate_value = self.task_gates[i][j](task_fea[j]).unsqueeze(1)
                task_fea[j] = torch.bmm(gate_value, mix_ouput).squeeze(1)
            if i != self.layers_num - 1:  # 最后一层不需要计算share expert 的输出
                gate_value = self.share_gates[i](task_fea[-1]).unsqueeze(1)
                mix_ouput = torch.cat(task_output_list + share_output, dim=1)
                task_fea[-1] = torch.bmm(gate_value, mix_ouput).squeeze(1)

        results = [
            self.tower[i](task_fea[i]).squeeze(1) for i in range(self.task_num)
        ]
        return results


class RecInterestQueryItemFeatureAlignment(nn.Module):
    def __init__(self, temp_init, hdim):
        super().__init__()
        self.infoNCE_loss = infoNCE(temp_init, hdim)

    # def filter_user_src_his(self, qry_his_emb, click_item_mask,
    #                         click_item_emb):
    #     qry_his_emb = qry_his_emb.unsqueeze(2).expand(-1, -1,
    #                                                   click_item_mask.size(2),
    #                                                   -1)

    #     src_his_query_emb = torch.masked_select(
    #         qry_his_emb.clone(),
    #         click_item_mask.unsqueeze(-1)).reshape(-1, qry_his_emb.size(-1))
    #     src_his_click_item_emb = torch.masked_select(click_item_emb.clone(), click_item_mask.unsqueeze(-1))\
    #         .reshape(-1, click_item_emb.size(-1))

        # return src_his_query_emb, src_his_click_item_emb

    def forward(self, align_loss_input, query_emb, item_emb):
        '''
        query_emb.shape
        torch.Size([16, 60, 64])
        q_click_item_emb.shape
        torch.Size([16, 60, 1, 64])        
        '''
        neg_item_emb, neg_query_emb = align_loss_input
        # neg_item_emb.shape: [neg_item_num, emb_dim]
        # neg_query_emb.shape: [neg_item_num, emb_dim]
        to_cal_query = query_emb.expand(item_emb.shape[0], -1)
        to_cal_item = item_emb[:, 0].to(query_emb.device)        # to_cal_query.shape: [src_his_num * batch_size, emb_dim]
        # to_cal_item.shape: [src_his_num * batch_size, emb_dim]
        align_loss = self.infoNCE_loss(to_cal_query,
                                       to_cal_item, neg_item_emb,
                                       neg_query_emb)
        return align_loss


class InterestContrast(nn.Module):
    def __init__(self, hid_dim, margin = 1.0):
        super().__init__()

        self.co_att = CoAttention(hid_dim)
        self.sigmoid = nn.Sigmoid()

        self.__init_Contrastive_Loss_(margin)

    def __init_Contrastive_Loss_(self, margin):

        self.trip_loss = nn.TripletMarginWithDistanceLoss(
            # default distance function is Eucdu
            margin = margin
        )
    
    def pooling(self, ele_set, mask):
        '''mean pooling for positives and negatives
        
        Args:
            ele_set: (batch, seq_len, dim)
            mask: (batch, seq_len)
        
        Returns:
            vector: (Batch, dim)
        '''
        mask = mask[:,None]
        len = mask.sum(-1)
        res = (ele_set * mask.transpose(2,1)).sum(1) / torch.max(len, torch.ones_like(len)) 
        return res #batch, dim
 
    def filter_pos_neg(self, w_rec, w_src, rec_mask=None, src_mask=None):
        '''select weights with gate 1/n. 
            weights >= 1/n -> positive
            weights < 1/n -> negative
        
        Args:
            w_rec, w_src: (B, 1, seq_len). attention scores
            rec_mask, src_mask: (B, seq_len), mask for padding values, "True" means padding

        Returns:
            mask_rec_pos, mask_rec_neg, mask_src_pos, mask_src_neg: (B, seq_len). positive and negative position masks for reco and src
        '''
        # batch, max_s_sequence          #batch, 1, max_s_seq
        w_rec, w_src = w_rec.squeeze(1), w_src.squeeze(1)
        rec_len, src_len = (~rec_mask).sum(-1, keepdim=True), (~src_mask).sum(-1, keepdim=True)
        rec_len, src_len = rec_len.expand(-1, w_rec.size(1)), src_len.expand(-1, w_src.size(1))

        large_gate_rec, large_gate_src = w_rec >= (1 / rec_len), w_src >= (1 / src_len)
        less_gate_rec, less_gate_src = w_rec < (1 / rec_len), w_src < (1 / src_len)

        # batch, max_s_sequence
        mask_rec_neg = less_gate_rec.masked_fill(rec_mask, 0.)
        mask_src_neg = less_gate_src.masked_fill(src_mask, 0.)

        mask_rec_pos = large_gate_rec.masked_fill(rec_mask, 0.)
        mask_src_pos = large_gate_src.masked_fill(src_mask, 0.)

        # Make sequences have at least one element for postive (negativeçc) samples
        # lenght == 1 or all weights equal 1/len may lead to all values False
        mask_rec_neg = torch.where(~mask_rec_neg.sum(-1, keepdim=True).bool(), ~rec_mask, mask_rec_neg)
        mask_rec_pos = torch.where(~mask_rec_pos.sum(-1, keepdim=True).bool(), ~rec_mask, mask_rec_pos)
        mask_src_neg = torch.where(~mask_src_neg.sum(-1, keepdim=True).bool(), ~src_mask, mask_src_neg)
        mask_src_pos = torch.where(~mask_src_pos.sum(-1, keepdim=True).bool(), ~src_mask, mask_src_pos)


        return mask_rec_pos, mask_rec_neg, mask_src_pos, mask_src_neg

    def forward(self, all_his, item, all_mask=None, return_loss=True):
        '''
        Args:
            all_his/item: (batch, seq/1, dim)
            all_mask: (batch, seq), True means padding
            return_loss(bool): whether return loss
        '''
        B, T, D = all_his.shape
        I = item.shape[1]
        all_his = all_his.repeat(I, 1, 1)
        all_mask = all_mask.repeat(I, 1)
        item = item.reshape(B * I, 1, D)
        item_mask = torch.zeros(B * I, 1).bool().to(all_mask.device)
        #get co-attention weights
        w_his, w_item = self.co_att(all_his, item, all_mask, item_mask) #B, 1, his_len.  B, 1, item_len

        #note: these masks set valuable elements with True, different with all_mask
        mask_his_pos, mask_his_neg, mask_item_pos, mask_item_neg =\
             self.filter_pos_neg(w_his, w_item, all_mask, item_mask) 

        if return_loss:
            anchor_rec = w_his @ all_his  #B, 1, dim
            
            his_loss = self.trip_loss(anchor_rec.squeeze(1), 
                                        self.pooling(all_his, mask_his_pos),
                                        self.pooling(all_his, mask_his_neg))

            return his_loss, mask_his_pos, mask_his_neg, mask_item_pos, mask_item_neg
        else:

            return mask_his_pos, mask_his_neg, mask_item_pos, mask_item_neg
        

class CoAttention(nn.Module):
    def __init__(self, embed_dim=100):
        super().__init__()

        self.embed_dim = embed_dim

        self.W1 = nn.parameter.Parameter( torch.rand((self.embed_dim, self.embed_dim)) )
        self.Wq = nn.parameter.Parameter( torch.randn((1, self.embed_dim)) )
        self.Wd = nn.parameter.Parameter( torch.randn((1, self.embed_dim)) )
        nn.init.xavier_normal_(self.W1)
        nn.init.xavier_normal_(self.Wq)
        nn.init.xavier_normal_(self.Wd)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

     
    def forward(self, query, doc, query_mask=None, doc_mask=None): 
        '''calculating co-attention scores which indicate the similarity of the common interests and the corresponding element

        Args:
            query, doc: (B, seq_len, dim)
            query_mask, doc_mask: (B, seq_len)

        Return: attention scores
            Aq, Ad: (B, 1, seq_len)
        '''
        query_trans = query.transpose(2, 1)
        doc_trans = doc.transpose(2, 1)
        L = self.tanh(torch.matmul(torch.matmul(query, self.W1), doc_trans)) # batch, max_s_query, max_s_doc
        L_trans = L.transpose(2, 1) # DWQ_T  batch, max_s_doc, max_s_query

        score_d = torch.matmul(torch.matmul(self.Wq, query_trans), L) #batch, 1, max_s_doc
        score_q = torch.matmul(torch.matmul(self.Wd, doc_trans), L_trans) #batch, 1, max_s_query

        score_d = score_d.masked_fill(doc_mask.unsqueeze(1), torch.tensor(-1e12))
        score_q = score_q.masked_fill(query_mask.unsqueeze(1), torch.tensor(-1e12))

        Aq = self.softmax(score_q) # [batchsize, 1, max_s_query]
        Ad = self.softmax(score_d) # [batchsize, 1, max_s_doc]

        return Aq, Ad
    

class TemporalAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1):
        super(TemporalAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)


    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        queries = queries.permute(0, 2, 1, 3)
        # queries: [batch_size, n_heads, seq_len, features]
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)
        attention = torch.matmul(queries, keys.transpose(-2, -1))
        attention /= (E ** 0.5)
        if attn_mask is not None:
            attention = attention.masked_fill(attn_mask == 0, -1e6)
        attention = F.softmax(attention, dim=-1)
        output = torch.matmul(attention, values)
        return output, attention


class TemporalAttentionLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(TemporalAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
class SequenceAttLayer(nn.Module):
    """Attention Layer. Get the representation of each user in the batch.

    Args:
        queries (torch.Tensor): candidate ads, [B, H], H means embedding_size * feat_num
        keys (torch.Tensor): user_hist, [B, T, H]
        keys_length (torch.Tensor): mask, [B]

    Returns:
        torch.Tensor: result
    """

    def __init__(
        self, mask_mat, att_hidden_size=(80, 40), activation='sigmoid', softmax_stag=False, return_seq_weight=True
    ):
        super(SequenceAttLayer, self).__init__()
        self.att_hidden_size = att_hidden_size
        self.activation = activation
        self.softmax_stag = softmax_stag
        self.return_seq_weight = return_seq_weight
        self.mask_mat = mask_mat
        self.att_mlp_layers = MLPLayers(self.att_hidden_size, activation='Sigmoid', bn=False)
        self.dense = nn.Linear(self.att_hidden_size[-1], 1)

    def forward(self, queries, keys, keys_length):
        embedding_size = queries.shape[-1]  # H
        hist_len = keys.shape[1]  # T
        queries = queries.repeat(1, hist_len)

        queries = queries.view(-1, hist_len, embedding_size)

        # MLP Layer
        input_tensor = torch.cat([queries, keys, queries - keys, queries * keys], dim=-1)
        output = self.att_mlp_layers(input_tensor)
        output = torch.transpose(self.dense(output), -1, -2)

        # get mask
        output = output.squeeze(1)
        mask = self.mask_mat.repeat(output.size(0), 1)
        mask = (mask >= keys_length.unsqueeze(1))

        # mask
        if self.softmax_stag:
            mask_value = -np.inf
        else:
            mask_value = 0.0

        output = output.masked_fill(mask=mask, value=torch.tensor(mask_value))
        output = output.unsqueeze(1)
        output = output / (embedding_size ** 0.5)

        # get the weight of each user's history list about the target item
        if self.softmax_stag:
            output = fn.softmax(output, dim=2)  # [B, 1, T]

        if not self.return_seq_weight:
            output = torch.matmul(output, keys)  # [B, 1, H]

        return output