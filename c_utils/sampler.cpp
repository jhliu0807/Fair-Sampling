#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <vector>

class InteractionSampler {
public:
    InteractionSampler(const std::map<int, std::set<int>>& record, const std::map<int, std::set<int>>& record_item, int num_user, int num_item)
        : record_(record), record_item_(record_item), num_user_(num_user), num_item_(num_item) {}

    std::vector<int> negative_sample(const std::vector<int>& user_batch) {
        std::vector<int> negative_samples;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, num_item_ - 1);

        for (int user_id : user_batch) {
            int item_id;
            do {
                item_id = dis(gen);
            } while (record_[user_id].count(item_id) > 0);
            negative_samples.push_back(item_id);
        }

        return negative_samples;
    }

    std::pair<std::vector<int>, std::vector<int>> sample(const std::vector<int>& user_batch) {
        std::vector<int> item_samples;
        std::vector<int> interactions;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, num_item_ - 1);

        for (int user_id : user_batch) {
            int item_id = dis(gen);
            item_samples.push_back(item_id);
            interactions.push_back(record_[user_id].count(item_id) > 0 ? 1 : 0);
        }

        return {item_samples, interactions};
    }

    std::pair<std::vector<int>, std::vector<int>> fair_sample(const std::vector<int>& user_batch, const std::vector<int>& item_batch) {
        std::vector<int> user_samples;
        std::vector<int> item_samples;
        int len = user_batch.size();

        for (int cur = 0; cur < len; cur++) {
            int u1 = user_batch[cur];
            int i1 = item_batch[cur];
            int delta = 1;
            int u2, i2;
            do {
                u2 = user_batch[(cur + delta) % len];
                i2 = item_batch[(cur + delta) % len];
                ++delta;
            } while (record_[u2].count(i1) > 0 || record_[u1].count(i2) > 0);
            user_samples.push_back(u2);
            item_samples.push_back(i2);
        }

        return {user_samples, item_samples};
    }

    std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> fair_sample_point(const std::vector<int>& user_batch, const std::vector<int>& item_batch, const std::vector<int>& label_batch) {
        std::vector<int> user_samples;
        std::vector<int> item_samples;
        std::vector<int> label_samples;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, num_user_ - 1);
        
        int len = user_batch.size();
        for (int cur = 0; cur < len; cur++) {
            int i2 = item_batch[cur];
            int u2;
            if (label_batch[cur] == 1) {
                do {
                    u2 = dis(gen);
                } while (record_[u2].count(i2) > 0);
                label_samples.push_back(0);
            } else {
                std::set<int> s = record_item_[i2];
                if (s.size() == 0) {
                    continue;
                }
                std::uniform_int_distribution<> dis_temp(0, s.size() - 1);
                auto it = s.begin();
                std::advance(it, dis_temp(gen));
                u2 = *it;
                label_samples.push_back(1);
            }
            user_samples.push_back(u2);
            item_samples.push_back(i2);
        }

        return std::make_tuple(user_samples, item_samples, label_samples);
    }

private:
    std::map<int, std::set<int>> record_;
    std::map<int, std::set<int>> record_item_;
    int num_user_;
    int num_item_;
};

PYBIND11_MODULE(sampler, m) {
    pybind11::class_<InteractionSampler>(m, "InteractionSampler")
        .def(pybind11::init<std::map<int, std::set<int>>, std::map<int, std::set<int>>, int, int>())
        .def("negative_sample", &InteractionSampler::negative_sample)
        .def("sample", &InteractionSampler::sample)
        .def("fair_sample", &InteractionSampler::fair_sample)
        .def("fair_sample_point", &InteractionSampler::fair_sample_point);
}
