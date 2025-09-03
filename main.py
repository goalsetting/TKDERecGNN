# %% ########################
# 主程序 main.py
#############################
from logging import getLogger

import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
# from torch import optim

# from modules import *
from modules.MainModel import DLGCN
# from modules.llm_enhancer import PaneLLMEnhancer
# from modules.pane_generator import DynamicPaneGenerator
# from modules.pane_model import PaneAwareModel

#ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIELv6KcoH5HJ3pPWiPtFJQNi5ZSfCQV78kQY+MgklCjd
config_dict = {
    # dataset config
    'num_panes': 1,
    'pane_update_interval': 20,  # 每3个epoch更新窗格
    'temperature': 0.7,
    'llm': {
      'cache_size': 100,
      'batch_size': 4,
      'max_desc_length': 128
    },
    'field_separator': "\t",  #指定数据集field的分隔符
    'seq_separator': " " ,  #指定数据集中token_seq或者float_seq域里的分隔符
    'USER_ID_FIELD': 'user_id' ,#指定用户id域
    'ITEM_ID_FIELD': 'item_id', #指定物品id域
    'RATING_FIELD': 'rating',  # 指定打分rating域
    'TIME_FIELD': 'timestamp',  # 指定时间域
    'load_col': {
        'inter': ['user_id', 'item_id', 'rating','timestamp']
    },
    'val_interval': {
        'rating': [4,5]  # Filter interactions with rating equal to 5
    },
    'NEG_PREFIX': 'neg_',   #指定负采样前缀
    'leave_one_out': False,
    # training settings
    'embedding_size': 32,
    'n_layers': 2,
    'reg_weight': 1e-5,
    'epochs': 150,  #训练的最大轮数
    'learner': 'adam', #使用的pytorch内置优化器
    'learning_rate': 0.002, #学习率
    'eval_step': 50, #每次训练后做evalaution的次数
    'stopping_step': 100, #控制训练收敛的步骤数，在该步骤数内若选取的评测标准没有什么变化，就可以提前停止了
    'group_by_user': True, #是否将一个user的记录划到一个组里，当eval_setting使用RO_RS的时候该项必须是True
    'split_ratio': {'RS': [0.7,0.1,0.2]}, #切分比例 ,"Precision","TailPercentage", "MRR"
    'metrics': ["Recall","TailRecall","HeadRecall","NDCG","TailNDCG","HeadNDCG","Hit","TailHit","HeadHit"], #评测标准
    'topk': [20], #评测标准使用topk，设置成10评测标准就是["Recall@10", "MRR@10", "NDCG@10", "Hit@10", "Precision@10"]
    'valid_metric': 'TailRecall@20', #选取哪c
    'eval_batch_size': 512,
    'train_batch_size': 512,
    'tail_ratio': 0.8,
    'gamma': 1,
    't': 1.8,
    'train_strategy': 'GODE',
    # evalution settings
    'eval_args': {
        'split': {
            'RS': [0.7, 0.1, 0.2]
        },
        'group_by': 'user',
        'order': 'RO'
        #        'mode': {'valid': 'uni100', 'test': 'uni100'}
    }
}

def update(dataset):
    # Calculate the threshold for head items (top 20%)
    total_items = dataset.item_counter
    item_counts = list(total_items.items())

    # Sort items based on counts (descending order)
    sorted_items = sorted(item_counts, key=lambda x: x[1], reverse=True)

    # Calculate the threshold for head items (top 20%)
    total_items = len(sorted_items)
    top_20_percent_threshold = int(total_items * 0.2)

    # Determine head items and tail items
    head_items = [item_id for item_id, count in sorted_items[:top_20_percent_threshold]]
    tail_items = [item_id for item_id, count in sorted_items[top_20_percent_threshold:]]

    # 更新或添加多个键值对
    config_dict.update({
        "tail_items": tail_items,
        "head_items": head_items  # 更新已存在的键
    })


def main():
    # 初始化配置
    # config = Config(model='PaneAwareModel', config_file_list=['config/pane_config.yaml'])

    name = 'ml-100k'

    config = Config(model='LightGCN', dataset=name, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    logger = getLogger()

    # logger.info(config)
    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    update(dataset)

    config = Config(model=DLGCN, dataset=name, config_dict=config_dict)
    init_logger(config)
    logger = getLogger()

    logger.info(config)


    train_data, valid_data, test_data = data_preparation(config, dataset)

    # 模型训练
    model = DLGCN(config, train_data._dataset).to(config['device'])
    trainer = Trainer(config, model)
    trainer.fit(train_data, valid_data)

    # 评估
    test_result = trainer.evaluate(test_data)
    print("Test Results:")
    print(f"Recall@20: {test_result['recall@20']:.4f}")
    print(f"NDCG@20: {test_result['ndcg@20']:.4f}")

    # # 初始化组件
    # pane_generator = DynamicPaneGenerator(config)
    # llm_enhancer = PaneLLMEnhancer(dataset, config)
    # model = PaneAwareModel(config, dataset).to(config['device'])
    # optimizer = optim.Adam(model.parameters(),
    #                        lr=0.02, weight_decay=1E10-4)
    #
    # # 训练循环
    # trainer = Trainer(config, model)
    # for epoch in range(config['train']['epochs']):
    #     # 定期更新窗格划分
    #     if epoch % config['pane_update_interval'] == 0:
    #         with torch.no_grad():
    #             user_emb = model.user_embedding.weight.data
    #             pane_labels, pane_members = pane_generator.update_panes(user_emb)
    #
    #             # 获取窗格嵌入
    #             pane_embs = []
    #             for pane_id, members in pane_members.items():
    #                 pane_items = train_data.inter_feat[train_data.inter_feat['user_id'].isin(members)]['item_id']
    #                 pane_emb = llm_enhancer.get_pane_embedding(pane_items.tolist())
    #                 pane_embs.append(pane_emb)
    #             pane_embs = torch.stack(pane_embs).to(config['device'])
    #
    #     # 训练步骤
    #     for batch in train_data:
    #         users = batch['user_id'].to(config['device'])
    #         items = batch['item_id'].to(config['device'])
    #         neg_items = batch['neg_item_id'].to(config['device'])
    #
    #         # 获取当前窗格嵌入
    #         batch_pane_labels = pane_labels[users.cpu().numpy()]
    #         batch_pane_embs = pane_embs[batch_pane_labels]
    #
    #         loss = model.calculate_loss({
    #             'user_id': users,
    #             'item_id': items,
    #             'neg_item_id': neg_items
    #         }, batch_pane_embs)
    #
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #
    #     # 评估
    #     valid_result = trainer.evaluate(valid_data)
    #     print(f"Epoch {epoch}: {valid_result}")


if __name__ == '__main__':
    main()