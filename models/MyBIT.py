import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
from utils import const

from .base_model import BaseModel
from .layers import FullyConnectedLayer, feature_align, PositionalEmbedding, PLE_layer


class MyBIT(BaseModel):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--num_layers', type=int, default=1)
        parser.add_argument('--num_heads', type=int, default=2)

        parser.add_argument('--q_i_cl_temp', type=float, default=0.5)
        parser.add_argument('--q_i_cl_weight', type=float, default=0.001)

        parser.add_argument('--his_cl_temp', type=float, default=0.1)
        parser.add_argument('--his_cl_weight', type=float, default=0.1)

        parser.add_argument('--pred_hid_units',
                            type=List,
                            default=[200, 80, 1])

        return BaseModel.parse_model_args(parser)

    def __init__(self, args):
        super().__init__(args)
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.batch_size = args.batch_size
        self.src_pos = PositionalEmbedding(const.max_src_session_his_len,
                                           self.item_size)
        self.rec_pos = PositionalEmbedding(const.max_rec_his_len,
                                           self.item_size)
        self.global_pos_emb = PositionalEmbedding(
            const.max_rec_his_len,
            self.item_size)

        self.rec_transformer = Transformer(emb_size=self.item_size,
                                           num_heads=self.num_heads,
                                           num_layers=self.num_layers,
                                           dropout=self.dropout)
        self.src_transformer = Transformer(emb_size=self.item_size,
                                           num_heads=self.num_heads,
                                           num_layers=self.num_layers,
                                           dropout=self.dropout)
        self.global_transformer = Transformer(emb_size=self.item_size,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_layers,
                                              dropout=self.dropout)

        self.q_i_cl_temp = args.q_i_cl_temp
        self.q_i_cl_weight = args.q_i_cl_weight
        if self.q_i_cl_weight > 0:
            self.query_item_alignment = True
            self.feature_alignment = feature_align(self.q_i_cl_temp,
                                                   self.item_size)

        self.his_cl_temp = args.his_cl_temp
        self.his_cl_weight = args.his_cl_weight
        if self.his_cl_weight > 0:
            self.rec_his_cl = TransAlign(batch_size=self.batch_size,
                                         hidden_dim=self.item_size,
                                         device=self.device,
                                         infoNCE_temp=self.his_cl_temp)
            self.src_his_cl = TransAlign(batch_size=self.batch_size,
                                         hidden_dim=self.item_size,
                                         device=self.device,
                                         infoNCE_temp=self.his_cl_temp)

        self.transformerDecoderLayer = nn.TransformerDecoderLayer(
            d_model=self.item_size,
            nhead=self.num_heads,
            dim_feedforward=self.item_size,
            dropout=self.dropout,
            batch_first=True)

        self.src_cross_fusion = nn.TransformerDecoder(
            self.transformerDecoderLayer, num_layers=self.num_layers)
        self.rec_cross_fusion = nn.TransformerDecoder(
            self.transformerDecoderLayer, num_layers=self.num_layers)

        self.rec_his_attn_pooling = Target_Attention(self.item_size,
                                                     self.item_size)
        self.src_his_attn_pooling = Target_Attention(self.item_size,
                                                     self.item_size)

        self.his_attn_pooling = Target_Attention(self.item_size,
                                                  self.item_size)
        self.q_attn_pooling = Target_Attention(self.item_size,
                                                  self.item_size)
        self.rec_query = torch.nn.parameter.Parameter(torch.randn(
            (1, self.query_size), requires_grad=True),
                                                      requires_grad=True)
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.batch_size = args.batch_size
        self.latent_dim = 64  # int type:the embedding size of lightGCN
        self.reg_weight = 1e-2  # float32 type: the weight decay for l2 normalization
        self.cross_layers = [128,64,32,16,8]  # list type: the list of hidden layers size
        self.source_crossunit_linear, self.source_crossunit_act \
            = self.cross_units([3 * self.latent_dim] + self.cross_layers)
        self.source_outputunit = nn.Sequential(
            nn.Linear(self.cross_layers[-1], 1),
            nn.Sigmoid()
        )
        nn.init.xavier_normal_(self.rec_query)
        self.target_inputunit = nn.Sequential(
            nn.Linear(4 * self.latent_dim, 3 * self.latent_dim),
         nn.ReLU()   
        )
        self.target_crossunit_linear, self.target_crossunit_act \
            = self.cross_units([3 * self.latent_dim] + self.cross_layers)
        

        self.target_outputunit = nn.Sequential(
            nn.Linear(self.cross_layers[-1], 1),
            nn.Sigmoid()
        )

        self.crossparas = self.cross_parameters([3 * self.latent_dim] + self.cross_layers)
        nn.init.xavier_normal_(self.rec_query)

        # self.hidden_unit = args.pred_hid_units

        # input_dim = 2 * self.item_size + self.user_size + self.query_size
        # self.ple_layer = PLE_layer(orig_input_dim=input_dim,
        #                            bottom_mlp_dims=[64],
        #                            tower_mlp_dims=[128, 64],
        #                            task_num=2,
        #                            shared_expert_num=4,
        #                            specific_expert_num=4,
        #                            dropout=self.dropout)
        # self.rec_fc_layer = FullyConnectedLayer(input_size=64,
        #                                         hidden_unit=self.hidden_unit,
        #                                         batch_norm=False,
        #                                         sigmoid=True,
        #                                         activation='relu',
        #                                         dropout=self.dropout)
        # self.src_fc_layer = FullyConnectedLayer(input_size=64,
        #                                         hidden_unit=self.hidden_unit,
        #                                         batch_norm=False,
        #                                         sigmoid=True,
        #                                         activation='relu',
        #                                         dropout=self.dropout)

        self.loss_fn = nn.BCELoss()
        self._init_weights()
        self.to(self.device)

    def src_feat_process(self, src_feat):
        query_emb, q_click_item_emb, click_item_mask = src_feat

        q_i_align_used = [query_emb, click_item_mask, q_click_item_emb]

        mean_click_item_emb = torch.sum(torch.mul(
            q_click_item_emb, click_item_mask.unsqueeze(-1)),
                                        dim=-2)  # batch, max_src_len, dim
        mean_click_item_emb = mean_click_item_emb / (torch.max(
            click_item_mask.sum(-1, keepdim=True),
            torch.ones_like(click_item_mask.sum(-1, keepdim=True))))
        query_his_emb = query_emb
        click_item_his_emb = mean_click_item_emb

        return query_his_emb + click_item_his_emb, q_i_align_used
    
    def cross_units(self, cross_layers):
        cross_modules_linear, cross_modules_act = [], []
        for i, (d_in, d_out) in enumerate(zip(cross_layers[:-1], cross_layers[1:])):
            cross_modules_linear.append(nn.Linear(d_in, d_out))
            cross_modules_act.append(nn.ReLU())
        return nn.ModuleList(cross_modules_linear), nn.ModuleList(cross_modules_act)

    def cross_parameters(self, cross_layers):
        cross_paras = []
        for i, (d_in, d_out) in enumerate(zip(cross_layers[:-1], cross_layers[1:])):
            para = nn.Linear(d_in, d_out, bias=False)
            cross_paras.append(para)
        return nn.ModuleList(cross_paras)


    def get_all_his_emb(self, all_his, all_his_type):
        # rec_his = torch.masked_fill(all_his, all_his_type != 1, 0)
        # rec_his_emb = self.session_embedding.get_query_emb(all_his)
        rec_his_emb = self.session_embedding.get_history_emb(all_his)
        # rec_his_emb = torch.masked_fill(rec_his_emb,
        #                                 (all_his_type != 1).unsqueeze(-1), 0)

        all_his_mask = torch.where(all_his == 0, 1, 0).bool()

        return rec_his_emb, all_his_mask, None

    def repeat_feat(self, feature_list, items_emb):
        repeat_feature_list = [
            torch.repeat_interleave(feat, items_emb.size(1), dim=0)
            for feat in feature_list
        ]
        items_emb = items_emb.reshape(-1, items_emb.size(-1))

        return repeat_feature_list, items_emb

    def mean_pooling(self, output, his_len):
        return torch.sum(output, dim=1) / his_len.unsqueeze(-1)

    def split_rec_src(self, all_his_emb, all_his_type):
        rec_his_emb = torch.masked_select(
            all_his_emb, (all_his_type == 1).unsqueeze(-1)).reshape(
                (all_his_emb.shape[0], const.max_rec_his_len,
                 all_his_emb.shape[2]))
        src_his_emb = torch.masked_select(
            all_his_emb, (all_his_type == 2).unsqueeze(-1)).reshape(
                (all_his_emb.shape[0], const.max_src_session_his_len,
                 all_his_emb.shape[2]))
        return rec_his_emb, src_his_emb

    def forward(self, user, all_his, all_his_type, items_emb, domain):
        user_emb = self.session_embedding.get_user_emb(user)

        all_his_emb, all_his_mask, _ = self.get_all_his_emb(
            all_his, all_his_type)

        # rec_his_mask = torch.masked_select(all_his_mask,
        #                                    (all_his_type == 1)).reshape(
        #                                        (all_his_emb.shape[0],
        #                                         const.max_rec_his_len))
        # src_his_mask = torch.masked_select(all_his_mask,
        #                                    (all_his_type == 2)).reshape(
        #                                        (all_his_emb.shape[0],
        #                                         const.max_src_session_his_len))

        all_his_emb_w_pos = all_his_emb + self.global_pos_emb(all_his_emb)

        # global_mask = all_his_type[:, :, None] == all_his_type[:, None, :]

        global_encoded = self.global_transformer(all_his_emb_w_pos,
                                                 all_his_mask)
        # src2rec, rec2src = self.split_rec_src(global_encoded, all_his_type)

        # rec_his_emb, src_his_emb = self.split_rec_src(all_his_emb,
        #                                               all_his_type)
        # rec_his_emb_w_pos = rec_his_emb + self.rec_pos(rec_his_emb)
        # src_his_emb_w_pos = src_his_emb + self.src_pos(src_his_emb)

        # rec2rec = self.rec_transformer(rec_his_emb_w_pos, rec_his_mask)
        # src2src = self.src_transformer(src_his_emb_w_pos, src_his_mask)

        # rec_fusion_decoded = self.rec_cross_fusion(
        #     tgt=rec2rec,
        #     memory=src2rec,
        #     tgt_key_padding_mask=rec_his_mask,
        #     memory_key_padding_mask=rec_his_mask)

        # src_fusion_decoded = self.src_cross_fusion(
        #     tgt=src2src,
        #     memory=rec2src,
        #     tgt_key_padding_mask=src_his_mask,
        #     memory_key_padding_mask=src_his_mask)

        I = items_emb.shape[1]
        # if domain == 'rec':
        feature_list = [
            global_encoded, user_emb
        ]
        repeat_feature_list, items_emb = self.repeat_feat(
            feature_list, items_emb)
        global_encoded, user_emb = repeat_feature_list
        all_his_mask = all_his_mask.repeat(I,1)
        his_item_fusion = self.his_attn_pooling(global_encoded, items_emb,
                                               all_his_mask)

        user_feats = [his_item_fusion, user_emb]

        return user_feats

    def inter_pred(self, user_feats, item_emb, domain, query_emb=None):
        assert domain in ["rec", "src"]
       
        #rec_interest, src_interest, user_emb = user_feats
        mixed_interest, user_emb = user_feats
        
        
        
        if domain == "rec":
            item_emb = item_emb.reshape(-1, item_emb.size(-1))
            #src_interest=self.his_attn_pooling(mixed_interest, self.rec_query.expand(item_emb.shape[0],-1)) 
            source_crossinput = torch.cat([mixed_interest, user_emb, item_emb], dim=1).to(self.device)
            # target_crossinput = torch.cat([user_emb, item_emb], dim=1).to(self.device)
            target_crossinput = torch.cat([mixed_interest,user_emb, item_emb,self.rec_query.expand(item_emb.shape[0], -1)], dim=1).to(self.device)
            target_crossinput = self.target_inputunit(target_crossinput)
            w_loss=[]
            w_loss=torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device='cuda')
            for i in range(len(self.source_crossunit_linear)):
                source_fc_module, source_act_module = self.source_crossunit_linear[i], self.source_crossunit_act[i]
                source_fc_module = source_fc_module
                source_act_module = source_act_module
                cross_para = self.crossparas[i].weight.t()
                target_fc_module, target_act_module = self.target_crossunit_linear[i], self.target_crossunit_act[i]
                target_fc_module = target_fc_module
                target_act_module = target_act_module

                source_crossoutput = source_fc_module(source_crossinput)
                source_cross_w = torch.mm(target_crossinput, cross_para)
                #w_loss = w_loss + wasserstein_distance(source_crossoutput,source_cross_w)
                #w_loss.append( F.kl_div((source_cross_w.softmax(dim=-1)+1e-10).log(), source_crossoutput.softmax(dim=-1), reduction='sum'))
                #w_loss=[source_cross_w,source_crossoutput]
                #source_crossoutput =source_crossoutput+source_cross_w
                w_loss=w_loss+F.kl_div((source_cross_w.softmax(dim=-1)+1e-10).log(), source_crossoutput.softmax(dim=-1), reduction='sum')

                source_crossoutput = source_act_module(source_crossoutput)

                target_crossoutput = target_fc_module(target_crossinput)
                target_crossoutput = target_crossoutput + torch.mm(source_crossinput, cross_para)

                target_crossoutput = target_act_module(target_crossoutput)

                source_crossinput = source_crossoutput
                target_crossinput = target_crossoutput

            source_out = self.source_outputunit(source_crossinput).squeeze()

            return source_out,w_loss


        elif domain == "src":
            if item_emb.dim() == 3:
                [query_emb], item_emb = self.repeat_feat([query_emb], item_emb)
            #src_interest=self.his_attn_pooling(mixed_interest, query_emb)
            source_crossinput = torch.cat([mixed_interest, user_emb, item_emb], dim=1).to(self.device)
            target_crossinput = torch.cat([mixed_interest, user_emb, item_emb,query_emb], dim=1).to(self.device)
            target_crossinput = self.target_inputunit(target_crossinput)
            #w_loss=[]
            w_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device='cuda')
            for i in range(len(self.target_crossunit_linear)):
                source_fc_module, source_act_module = self.source_crossunit_linear[i], self.source_crossunit_act[i]
                source_fc_module = source_fc_module
                source_act_module = source_act_module
                cross_para = self.crossparas[i].weight.t()
                target_fc_module, target_act_module = self.target_crossunit_linear[i], self.target_crossunit_act[i]
                target_fc_module = target_fc_module
                target_act_module = target_act_module

                source_crossoutput = source_fc_module(source_crossinput)
                source_crossoutput =source_crossoutput+ torch.mm(target_crossinput, cross_para)

                source_crossoutput = source_act_module(source_crossoutput)

                target_crossoutput = target_fc_module(target_crossinput)
                target_cross_w = torch.mm(source_crossinput, cross_para)
                #w_loss.append( F.kl_div((target_cross_w.softmax(dim=-1)+1e-10).log(), target_crossoutput.softmax(dim=-1), reduction='sum'))
                #w_loss=[target_cross_w,target_crossoutput]
                #target_crossoutput = target_crossoutput+target_cross_w
                w_loss=w_loss+F.kl_div((target_cross_w.softmax(dim=-1)+1e-10).log(), target_crossoutput.softmax(dim=-1), reduction='sum')
                target_crossoutput = target_act_module(target_crossoutput)

                source_crossinput = source_crossoutput
                target_crossinput = target_crossoutput

            target_out = self.target_outputunit(target_crossinput).squeeze()
            
            return target_out,w_loss

    def rec_loss(self, inputs):

        user, all_his, pos_item, neg_items, title, _9000, _9001, _9002, _9003, _9004 = inputs['user'], inputs['all_his_1200'], inputs['item'], inputs['neg_items'], inputs["title"], inputs["_9000"], inputs["_9001"], inputs["_9002"], inputs["_9003"], inputs["_9004"]

        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)
        title_with_neg = torch.cat([title.unsqueeze(1), torch.zeros((items.shape[0], items.shape[1] - 1, title.shape[-1]), device=items.device)], dim=1)
        item_get_command = {
            "item_id": items.int(),
            "caption": title_with_neg.int()
        }
        items_emb = self.session_embedding.get_item_emb(item_get_command)
        batch_size = items_emb.size(0)
        user_get_command = {
            "user_id": user.int(),
            "_9000": _9000.int(),
            "_9001": _9001.int(),
            "_9002": _9002.int(),
            "_9003": _9003.int(),
            "_9004": _9004.int(),
        }
        user_feats = self.forward(user_get_command,
                                                               all_his.int(),
                                                               None,
                                                               items_emb,
                                                               domain='rec')

        #logits,w_loss = self.inter_pred(user_feats, items_emb, domain="rec").reshape(
        #    (batch_size, -1))
        logits,w_loss = self.inter_pred(user_feats, items_emb, domain="rec")
        logits = logits.reshape(
            (batch_size, -1))
        labels = torch.zeros_like(logits, dtype=torch.float32)
        labels[:, 0] = 1.0

        logits = logits.reshape((-1, ))
        labels = labels.reshape((-1, ))

        total_loss = self.loss_fn(logits, labels)
        loss_dict = {}
        loss_dict['click_loss'] = total_loss.clone()
        total_loss +=  w_loss
        # if self.q_i_cl_weight > 0:
        #     align_neg_item, align_neg_query = inputs['align_neg_item'], inputs[
        #         'align_neg_query']
        #     query_emb, click_item_mask, q_click_item_emb = q_i_align_used

        #     align_neg_items_emb = self.session_embedding.get_item_emb(
        #         align_neg_item)
        #     align_neg_querys_emb = self.session_embedding.get_query_emb(
        #         align_neg_query)
        #     align_loss = self.feature_alignment(
        #         [align_neg_items_emb, align_neg_querys_emb], query_emb,
        #         click_item_mask, q_click_item_emb)
        #     loss_dict['q_i_cl_loss'] = align_loss.clone()

        #     total_loss += self.q_i_cl_weight * align_loss

        # if self.his_cl_weight > 0:
            # src2rec, rec2rec, rec_his_mask,\
            #     rec2src, src2src, src_his_mask = his_cl_used
            # rec_his_cl_loss = self.rec_his_cl(src2rec, rec2rec, rec_his_mask)

            # src_his_cl_loss = self.src_his_cl(rec2src, src2src, src_his_mask)

            # his_cl_loss = rec_his_cl_loss + src_his_cl_loss
            # loss_dict['his_cl_loss'] = his_cl_loss.clone()

            # total_loss += self.his_cl_weight * his_cl_loss

        loss_dict['total_loss'] = total_loss

        return loss_dict

    def rec_predict(self, inputs):
        user, all_his, pos_item, neg_items, title, _9000, _9001, _9002, _9003, _9004 = inputs['user'], inputs['all_his_1200'], inputs['item'], inputs['neg_items'], inputs["title"], inputs["_9000"], inputs["_9001"], inputs["_9002"], inputs["_9003"], inputs["_9004"]

        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)
        title_with_neg = torch.cat([title.unsqueeze(1), torch.zeros((items.shape[0], items.shape[1] - 1, title.shape[-1]), device=items.device)], dim=1)
        item_get_command = {
            "item_id": items.int(),
            "caption": title_with_neg.int()
        }
        items_emb = self.session_embedding.get_item_emb(item_get_command)
        batch_size = items_emb.size(0)
        user_get_command = {
            "user_id": user.int(),
            "_9000": _9000.int(),
            "_9001": _9001.int(),
            "_9002": _9002.int(),
            "_9003": _9003.int(),
            "_9004": _9004.int(),
        }
        user_feats = self.forward(user_get_command,
                                all_his.int(),
                                None,
                                items_emb,
                                domain='rec')

        #logits = self.inter_pred(user_feats, items_emb, domain="rec").reshape(
        #    (batch_size, -1))
        logits,w_loss = self.inter_pred(user_feats, items_emb, domain="rec")
        logits = logits.reshape(
            (batch_size, -1))
        return logits

    def src_loss(self, inputs):
        user, all_his, pos_item, neg_items, title, _9000, _9001, _9002, _9003, _9004 = inputs['user'], inputs['all_his_1200'], inputs['item'], inputs['neg_items'], inputs["title"], inputs["_9000"], inputs["_9001"], inputs["_9002"], inputs["_9003"], inputs["_9004"]


        query = inputs['query']
        query_emb = self.session_embedding.get_query_emb(query)
        
        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)
        title_with_neg = torch.cat([title.unsqueeze(1), torch.zeros((items.shape[0], items.shape[1] - 1, title.shape[-1]), device=items.device)], dim=1)
        item_get_command = {
            "item_id": items.int(),
            "caption": title_with_neg.int()
        }
        items_emb = self.session_embedding.get_item_emb(item_get_command)
        batch_size = items_emb.size(0)
        user_get_command = {
            "user_id": user.int(),
            "_9000": _9000.int(),
            "_9001": _9001.int(),
            "_9002": _9002.int(),
            "_9003": _9003.int(),
            "_9004": _9004.int(),
        }

        user_feats = self.forward(user_get_command,
                                                               all_his.int(),
                                                               None,
                                                               items_emb,
                                                               domain='rec')

        logits,w_loss = self.inter_pred(user_feats,
                                 items_emb,
                                 domain="src",
                                 query_emb=query_emb)
        logits = logits.reshape((batch_size, -1))
        labels = torch.zeros_like(logits, dtype=torch.float32)
        labels[:, 0] = 1.0

        logits = logits.reshape((-1, ))
        labels = labels.reshape((-1, ))

        total_loss = self.loss_fn(logits, labels)
        loss_dict = {}
        loss_dict['click_loss'] = total_loss.clone()
        total_loss += w_loss
        # if self.q_i_cl_weight > 0:
        #     align_neg_item, align_neg_query = inputs['align_neg_item'], inputs[
        #         'align_neg_query']
        #     query_emb, click_item_mask, q_click_item_emb = q_i_align_used

        #     align_neg_items_emb = self.session_embedding.get_item_emb(
        #         align_neg_item)
        #     align_neg_querys_emb = self.session_embedding.get_query_emb(
        #         align_neg_query)
        #     align_loss = self.feature_alignment(
        #         [align_neg_items_emb, align_neg_querys_emb], query_emb,
        #         click_item_mask, q_click_item_emb)
        #     loss_dict['q_i_cl_loss'] = align_loss.clone()

        #     total_loss += self.q_i_cl_weight * align_loss

        # if self.his_cl_weight > 0:
        #     src2rec, rec2rec, rec_his_mask,\
        #         rec2src, src2src, src_his_mask = his_cl_used

        #     rec_his_cl_loss = self.rec_his_cl(src2rec, rec2rec, rec_his_mask)

        #     src_his_cl_loss = self.src_his_cl(rec2src, src2src, src_his_mask)

        #     his_cl_loss = rec_his_cl_loss + src_his_cl_loss
        #     loss_dict['his_cl_loss'] = his_cl_loss.clone()

        #     total_loss += self.his_cl_weight * his_cl_loss

        loss_dict['total_loss'] = total_loss

        return loss_dict

    def src_predict(self, inputs):
        user, all_his, pos_item, neg_items, title, _9000, _9001, _9002, _9003, _9004 = inputs['user'], inputs['all_his_1200'], inputs['item'], inputs['neg_items'], inputs["title"], inputs["_9000"], inputs["_9001"], inputs["_9002"], inputs["_9003"], inputs["_9004"]

        query = inputs['query']
        query_emb = self.session_embedding.get_query_emb(query)

        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)
        title_with_neg = torch.cat([title.unsqueeze(1), torch.zeros((items.shape[0], items.shape[1] - 1, title.shape[-1]), device=items.device)], dim=1)
        item_get_command = {
            "item_id": items.int(),
            "caption": title_with_neg.int()
        }
        items_emb = self.session_embedding.get_item_emb(item_get_command)
        batch_size = items_emb.size(0)
        user_get_command = {
            "user_id": user.int(),
            "_9000": _9000.int(),
            "_9001": _9001.int(),
            "_9002": _9002.int(),
            "_9003": _9003.int(),
            "_9004": _9004.int(),
        }

        user_feats = self.forward(user_get_command,
                                                               all_his.int(),
                                                               None,
                                                               items_emb,
                                                               domain='rec')

        logits,w_loss = self.inter_pred(user_feats,
                                 items_emb,
                                 domain="src",
                                 query_emb=query_emb)
        logits = logits.reshape((batch_size, -1))
        return logits


class Target_Attention(nn.Module):
    def __init__(self, hid_dim1, hid_dim2):
        super().__init__()

        self.W = nn.Parameter(torch.randn((1, hid_dim1, hid_dim2)))
        nn.init.xavier_normal_(self.W)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, seq_emb, target, mask=None):
        score = torch.matmul(seq_emb, self.W)
        score = torch.matmul(score, target.unsqueeze(-1))

        all_score = score.masked_fill(mask.unsqueeze(-1), torch.tensor(-1e16))
        #if mask:
        #    score = score.masked_fill(mask.unsqueeze(-1), torch.tensor(-1e16))
        all_weight = self.softmax(all_score.transpose(-2, -1))
        all_vec = torch.matmul(all_weight, seq_emb).squeeze(1)

        return all_vec


class TransAlign(nn.Module):
    def __init__(self, batch_size, hidden_dim, device, infoNCE_temp) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.device = device

        self.infoNCE_temp = nn.Parameter(torch.ones([]) * infoNCE_temp)
        self.weight_matrix = nn.Parameter(torch.randn(
            (hidden_dim, hidden_dim)))
        nn.init.xavier_normal_(self.weight_matrix)

        self.cl_loss_func = nn.CrossEntropyLoss()
        self.mask_default = self.mask_correlated_samples(self.batch_size)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool, device=self.device)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, same_his: torch.Tensor, diff_his: torch.Tensor,
                his_mask: torch.Tensor):
        same_his_emb = same_his.masked_fill(his_mask.unsqueeze(2), 0)
        same_his_sum = same_his_emb.sum(dim=1)
        same_his_mean = same_his_sum / \
            (~his_mask).sum(dim=1, keepdim=True)

        diff_his_emb = diff_his.masked_fill(his_mask.unsqueeze(2), 0)
        diff_his_sum = diff_his_emb.sum(dim=1)
        diff_his_mean = diff_his_sum / \
            (~his_mask).sum(dim=1, keepdim=True)

        batch_size = same_his_mean.size(0)
        N = 2 * batch_size

        z = torch.cat([same_his_mean.squeeze(),
                       diff_his_mean.squeeze()],
                      dim=0)
        sim = torch.mm(torch.mm(z, self.weight_matrix), z.T)
        sim = torch.tanh(sim) / self.infoNCE_temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        info_nce_loss = self.cl_loss_func(logits, labels)

        return info_nce_loss


class Transformer(nn.Module):
    def __init__(self, emb_size, num_heads, num_layers, dropout) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.transformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=num_heads,
            dim_feedforward=emb_size,
            dropout=dropout,
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformerEncoderLayer, num_layers=num_layers)

    def forward(self,
                his_emb: torch.Tensor,
                src_key_padding_mask: torch.Tensor,
                src_mask: torch.Tensor = None):
        if src_mask is not None:
            src_mask_expand = src_mask.unsqueeze(1).expand(
                (-1, self.num_heads, -1, -1)).reshape(
                    (-1, his_emb.size(1), his_emb.size(1)))
            his_encoded = self.transformer_encoder(
                src=his_emb,
                src_key_padding_mask=src_key_padding_mask,
                mask=src_mask_expand)
        else:
            his_encoded = self.transformer_encoder(
                src=his_emb, src_key_padding_mask=src_key_padding_mask)

        return his_encoded
