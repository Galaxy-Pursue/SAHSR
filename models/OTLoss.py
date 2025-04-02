import torch
import ot
import numpy as np 


def sinkhorn_knopp(M, u, v, K):
    M = M / np.max(M)  # 对矩阵 M 进行归一化
    P = np.diag(u) @ M @ np.diag(v)
    epsilon = 1e-10  # 防止除零的极小值
    for i in range(K):
        row_sums = np.sum(P, axis=1) + epsilon  # 防止除零
        P = P / row_sums[:, np.newaxis]
        P = P * u[:, np.newaxis]
        
        col_sums = np.sum(P, axis=0) + epsilon  # 防止除零
        P = P / col_sums[np.newaxis, :]
        P = P * v[np.newaxis, :]
    return P

    
    
class OTLoss(torch.nn.Module):
    def __init__(self, lambda_reg=0.1):
        super(OTLoss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, features_mod1, features_mod2, labels):
        unique_labels = torch.unique(labels)
        ot_loss = 0.0
        for label in unique_labels:
            mask = (labels == label)
            X = features_mod1[mask].cpu()
            Y = features_mod2[mask].cpu()
            if X.size(0) > 0 and Y.size(0) > 0:
                M = ot.dist(X.detach().numpy(), Y.detach().numpy())
                a = torch.ones(X.size(0)).numpy() / X.size(0)
                b = torch.ones(Y.size(0)).numpy() / Y.size(0)
                # ot_loss += ot.sinkhorn(a, b, M, self.lambda_reg).sum()
                ot_loss += sinkhorn_knopp(M, a, b,  20).sum()
                # # 使用 sinkhorn_stabilized 进行稳定化处理
                # ot_matrix = ot.sinkhorn_stabilized(a, b, M, self.lambda_reg)
                # print(f"OT Matrix: {ot_matrix}")  # 调试输出
                # ot_loss += ot_matrix.sum()
        return torch.tensor(ot_loss)

# 数据加载和特征提取
# 假设features_mod1和features_mod2是从两个不同模态提取的特征
# labels是对应的ID标签

if __name__ == "__main__":

    batch_size = 64
    feature_dim = 128
    features_mod1 = torch.rand((batch_size, feature_dim))
    features_mod2 = torch.rand((batch_size, feature_dim))
    # labels = torch.randint(0, 10, (batch_size,))
    print(torch.equal(features_mod1,features_mod1))
    labels = torch.tensor([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7])




    criterion = OTLoss(lambda_reg=1e-3)

    loss = criterion(features_mod1, features_mod1, labels)
    print(loss)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # for epoch in range(num_epochs):
    #     for data in dataloader:
    #         features_mod1, features_mod2, labels = model(data)
    #         loss = criterion(features_mod1, features_mod2, labels)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
