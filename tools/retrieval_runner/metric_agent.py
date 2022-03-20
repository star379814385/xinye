import faiss
import numpy as np




class MetricAgent(object):
    def __init__(self, use_gpu=False):
        super(MetricAgent, self).__init__()
        self.reset_gallery()
        self.reset_dim_process()
        self.reset_fea_enhancer()
        self.use_gpu = use_gpu

    def init_finder(self):
        cpu_index = faiss.IndexFlatL2(self.d)
        # cpu_index = faiss.IndexFlatIP(self.d)
        return faiss.index_cpu_to_all_gpus(cpu_index) if self.use_gpu else cpu_index

    def reset_gallery(self):
        self.gallery = None
        self.gallery_label = None
        self.d = None
        self.finder = None

    def set_gallery(self, gallery, gallery_label):
        assert gallery.shape[0] == gallery_label.shape[0]

        # dim_process
        if self.dim_process is not None:
            for dp in self.dim_process:
                gallery = dp(gallery)

        gallery_fe = [gallery]
        gallery_label_fe = [gallery_label]

        if self.features_enhancer is not None:
            for fe in self.features_enhancer:
                gallery_fe.append(fe(gallery))
                gallery_label_fe.append(gallery_label)
        gallery = np.concatenate(gallery_fe, axis=0)
        gallery_label = np.concatenate(gallery_label_fe, axis=0)

        self.gallery = gallery
        self.gallery_label = gallery_label
        self.d = self.gallery.shape[-1]
        # set finder
        self.finder = self.init_finder()
        self.finder.add(gallery)
        print("features dim:{}, gallery length:{}".format(self.d, self.gallery.shape[0]))

    def set_dim_process(self, dim_process):
        self.dim_process = dim_process
        print("using dim_process as follow:")
        print([d.__class__.__name__ for d in self.dim_process])

    def reset_dim_process(self):
        self.dim_process = None

    def set_fea_enhancer(self, fea_enhancer):
        self.features_enhancer = fea_enhancer
        print("using fea_enhancer as follow:")
        print([d.__class__.__name__ for d in self.features_enhancer])

    def reset_fea_enhancer(self):
        self.features_enhancer = None
        # print("--------------------")

    def run_metric(self, query_vector):
        query_vector = query_vector.astype(np.float32)
        # dim_process
        if self.dim_process is not None:
            for dp in self.dim_process:
                query_vector = dp(query_vector)

        k = 1  # we want to see 10 nearest neighbors
        D, I = self.finder.search(query_vector, k)  # actual search

        # 匹配策略
        # 1.取匹配相似度最大的图像类别
        # pred_cls_ids = np.array(list(map(lambda x: self.gallery_label[x], I[:, 0])), dtype=np.int32)
        # 2.top k
        pred_cls_ids = np.array(list(map(lambda x: self.gallery_label[x], I)), dtype=np.int32)
        pred_cls_ids = np.array(list(map(lambda x: np.argmax(np.bincount(x)), pred_cls_ids)), dtype=np.int32)
        return pred_cls_ids
