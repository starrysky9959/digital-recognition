/**
 * @author  starrysky
 * @date    2020/08/16
 * @details LibTorch调用PyTorch已经训练好的模型, 对传入的OpenCV的灰度图进行分类
 */

#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <opencv2/tracking.hpp>


class DigitalRecognition {
private:
    torch::jit::script::Module module;
    torch::Device device;
    const int IMAGE_COLS = 28;
    const int IMAGE_ROWS = 28;
public:
    /**
     * 默认使用CPU，可通过标志位开启使用GPU
     * @param use_cuda 是否使用GPU
     * @param model_path 模型文件路径
     */
    explicit DigitalRecognition(bool use_cuda = false,
                                const std::string &model_path = "../model/model.pt") : device(torch::kCPU) {
        if ((use_cuda) && (torch::cuda::is_available())) {
            std::cout << "CUDA is available! Training on GPU." << std::endl;
            device = torch::kCUDA;
        }
        module = torch::jit::load(model_path, device);
    }

    /**
     * 单张图片分类器
     * @param img 图片，cv::Mat类型
     * @return 分类结果
     */
    int matToDigital(cv::Mat &img) {
        // 正则化
        img.convertTo(img, CV_32FC1, 1.0f / 255.0f);

        // 模型用的是 28*28 的单通道灰度图
        cv::resize(img, img, cv::Size(IMAGE_COLS, IMAGE_ROWS));

        // 将 OpenCV 的 Mat 转换为 Tensor, 注意两者的数据格式
        // OpenCV: H*W*C 高度, 宽度, 通道数
        auto input_tensor = torch::from_blob(img.data, {1, IMAGE_COLS, IMAGE_ROWS, 1});

        // Tensor: N*C*H*W 数量, 通道数, 高度, 宽度
        // 数字表示顺序
        input_tensor = input_tensor.permute({0, 3, 1, 2}).to(device);

        // 添加数据
        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(input_tensor);

        // 模型计算
        at::Tensor output = module.forward(inputs).toTensor();
        std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/7) << '\n';

        // 输出分类的结果
        int ans = output.argmax(1).item().toInt();
        std::cout << "当前机器人编号: " << ans << std::endl;

        return ans;
    }
};

int main() {
    DigitalRecognition digitalRecognition;
    cv::Mat img = cv::imread("../image/1.jpg", CV_8UC1);
    digitalRecognition.matToDigital(img);
    return 0;
}
