#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <string>
#include <vector>

using cv::Mat;
using std::vector;
using std::string;
using std::cout;
using std::endl;

static const vector<string> class_name = {"animals", "cat", "chicken", "cow", "dog", "fox", "goat", "horse", "person", "racoon", "skunk"};

void print_result(const Mat &result, float conf = 0.3, int len_data = 16)
{
    // cout << result.total() <<endl;
    float *pdata = (float *)result.data;
    for(int i = 0; i < result.total() / len_data; i++)
    {
        if(pdata[4] > conf){
            for(int j = 0; j < len_data; j++)
            {
                cout << pdata[j] <<" ";
            }
            cout << endl;
        }
        pdata += len_data;
    }
    return;
}

vector<vector<float>> get_info(const Mat &result, float conf = 0.3, int len_data = 16)
{
    // cout << result.total() <<endl;
    float *pdata = (float *)result.data;
    vector<vector<float>> info;
    for(int i = 0; i < result.total() / len_data; i++)
    {
        vector<float> info_line;
        if(pdata[4] > conf){
            for(int j = 0; j < len_data; j++)
            {
                // cout << pdata[j] <<" ";
                info_line.push_back(pdata[j]);
            }
            // cout << endl;
            info.push_back(info_line);
        }
        pdata += len_data;
    }
    return info;
}

void info_simplify(vector<vector<float>> &info)
{
    for(auto i = 0; i < info.size(); i++)
    {
        info[i][5] = (std::max_element(info[i].cbegin() + 5, info[i].cend()) - (info[i].cbegin() + 5));
        info[i].resize(6);
        float x = info[i][0];
        float y = info[i][1];
        float w = info[i][2];
        float h = info[i][3];
        info[i][0] = x - w / 2.0;
        info[i][1] = y - h / 2.0;
        info[i][2] = x + w / 2.0;
        info[i][3] = y + h / 2.0;
    }
}

vector<vector<vector<float>>> split_info(vector<vector<float>> &info)
{
    vector<vector<vector<float>>> info_split;
    vector<int> class_id;
    for(auto i = 0; i < info.size(); i++)
    {
        if(std::find(class_id.begin(), class_id.end(), (int)info[i][5]) == class_id.end())
        {
            class_id.push_back((int)info[i][5]);
            vector<vector<float>> info_;
            info_split.push_back(info_);
        }
        info_split[std::find(class_id.begin(), class_id.end(), (int)info[i][5]) - class_id.begin()].push_back(info[i]);
    }
    return info_split;
}

void nms(vector<vector<float>> &info, float iou = 0.4)
{
    int counter = 0;
    vector<vector<float>> result_info;
    while(counter < info.size())
    {
        result_info.clear();
        float x1 = 0;
        float y1 = 0;
        float x2 = 0;
        float y2 = 0;
        std::sort(info.begin(), info.end(), [](vector<float> p1, vector<float> p2){ return p1[4] > p2[4]; });
        for(auto i = 0; i < info.size(); i++)
        {
            if(i < counter)
            {
                result_info.push_back(info[i]);
                continue;
            }
            if(i == counter)
            {
                x1 = info[i][0];
                y1 = info[i][1];
                x2 = info[i][2];
                y2 = info[i][3];
                result_info.push_back(info[i]);
                continue;
            }
            if(info[i][0] > x2 || info[i][2] < x1 || info[i][1] > y2 || info[i][3] < y1)
            {
                result_info.push_back(info[i]);
            }
            else
            {
                float over_x1 = std::max(x1, info[i][0]);
                float over_y1 = std::max(y1, info[i][1]);
                float over_x2 = std::min(x2, info[i][2]);
                float over_y2 = std::min(y2, info[i][3]);
                float s_over = (over_x2 - over_x1) * (over_y2 - over_y1);
                float s_total = (x2 - x1) * (y2 - y1) + (info[i][2] - info[i][0]) * (info[i][3] - info[i][1]) - s_over;
                if(s_over/ s_total < iou)
                {
                    result_info.push_back(info[i]);
                }
            }
        }
        info = result_info;
        counter += 1;
    }
}

void print_info(const vector<vector<float>> &info)
{
    for(auto i = 0; i < info.size(); i++)
    {
        for(auto j = 0; j < info[i].size(); j++)
        {
            cout << info[i][j] << " ";
        }
        cout << endl;
    }
}

void draw_box(Mat &img, const vector<vector<float>> &info)
{
    for(auto i = 0; i < info.size(); i++)
    {
        cv::rectangle(img, cv::Point(info[i][0], info[i][1]), cv::Point(info[i][2], info[i][3]), cv::Scalar(0, 255, 0));
        string label;
        label += class_name[(int)info[i][5]];
        label += ":";
        std::stringstream oss;
        oss << info[i][4];
        label += oss.str();
        cv::putText(img, label, cv::Point(info[i][0], info[i][1]), 1, 2, cv::Scalar(0, 255, 0));
    }
}

int main() {

    // if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
    //     cout << "CUDA is not available!" << endl;
    //     return -1;
    // } else {
    //     cout << "CUDA is available!" << endl;
    // }

    // cv::dnn::Net net = cv::dnn::readNetFromONNX("best.onnx");
    // Mat img = cv::imread("2.jpg");
    // cv::resize(img, img, cv::Size(640, 640));
    // Mat blob = cv::dnn::blobFromImage(img, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true);
    // net.setInput(blob);
    // vector<Mat> netoutput;
    // vector<string> out_name = {"output"};
    // net.forward(netoutput, out_name);
    // Mat result = netoutput[0];
    // // print_result(result);
    // vector<vector<float>> info = get_info(result);
    // info_simplify(info);
    // // print_info(info);
    // vector<vector<vector<float>>> info_split = split_info(info);
    // cout << "info_split:" <<endl;
    // print_info(info_split[0]);
    // cout << info.size() << " " << info[0].size() << endl;
    // cout << "nms:" <<endl;
    // nms(info_split[0]);
    // print_info(info_split[0]);
    // draw_box(img, info_split[0]);
    // cv::imshow("test", img);
    // cv::waitKey(0);
    // 1. 加载ONNX模型
    cv::dnn::Net net = cv::dnn::readNetFromONNX("best.onnx");

    // 2. 打开摄像头
    string video_path = R"(E:\code\YOLO\yolov5-6.0_cpp\1.avi)";
    cv::VideoCapture cap(video_path); // 0 默认是第一个摄像头
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to access the camera!" << endl;
        return -1;
    }

    // 3. 设置输入图像大小
    cv::Size input_size(640, 640); // 输入尺寸


    // // 获取源视频的帧率
    // double fps = cap.get(cv::CAP_PROP_FPS);
    // std::cout << "Source video FPS: " << fps << std::endl;

    // int frame_count = 0;
    // double fps = 0.0;
    // clock_t start_time = clock();  // 开始计时

    // // 获取视频的宽度和高度
    // int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    // int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // 创建 VideoWriter 对象，指定输出视频的文件名、编码格式、帧率等
    cv::VideoWriter outVideo("E:\\code\\YOLO\\yolov5-6.0_cpp\\output_video.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 4.0, input_size);
    
    if (!outVideo.isOpened()) {
        std::cerr << "Error: Could not open output video file!" << std::endl;
        return -1;
    }


    while (true) {
        Mat frame;
        cap >> frame; // 从摄像头读取一帧
        if (frame.empty()) {
            std::cerr << "Error: Unable to capture an image!" << endl;
            break;
        }

        // // 手动计算 FPS
        // frame_count++;

        // // 每处理一秒钟更新一次FPS
        // clock_t current_time = clock();
        // double elapsed_time = double(current_time - start_time) / CLOCKS_PER_SEC;
        
        // if (elapsed_time >= 1.0) {  // 每秒钟更新一次FPS
        //     fps = frame_count / elapsed_time;  // 计算FPS
        //     frame_count = 0;  // 重置帧数
        //     start_time = current_time;  // 重置计时器
        // }

        // cout << "Source video FPS: " << fps << endl;

        // 4. 图像预处理
        Mat img;
        cv::resize(frame, img, input_size); // 调整图像大小为640x640
        Mat blob = cv::dnn::blobFromImage(img, 1.0 / 255.0, input_size, cv::Scalar(), true); // 归一化处理
        net.setInput(blob);

        // 5. 前向推理
        vector<Mat> netoutput;
        vector<string> out_name = {"output"};
        net.forward(netoutput, out_name);

        // 6. 获取推理结果并处理
        Mat result = netoutput[0];
        vector<vector<float>> info = get_info(result); // 获取推理信息
        info_simplify(info); // 简化信息

        // 7. 分割信息（比如分割成不同类别或框）
        vector<vector<vector<float>>> info_split = split_info(info);
        cout << "info_split size: " << info_split.size() << endl;

        // 8. 执行非最大值抑制（NMS）
        for(auto i = 0; i < info_split.size(); i++)
            nms(info_split[i]);

        // 9. 绘制检测框
        for(auto i = 0; i < info_split.size(); i++)
            draw_box(img, info_split[i]);


        // 将处理后的帧保存到视频文件中
        outVideo.write(img);  // 这里保存原始帧，可以选择保存处理后的帧


        // 10. 显示结果
        cv::imshow("Real-time Object Detection", img);

        // 11. 检查是否按下退出键
        if (cv::waitKey(1) == 27) { // 按下'Esc'键退出
            break;
        }
    }

    cap.release(); // 释放摄像头
    outVideo.release();
    cv::destroyAllWindows(); // 关闭所有OpenCV窗口
    cout << "hello" << endl;
    return 0;
}

// #include <opencv2/opencv.hpp>
// #include <iostream>

// int main() {
//     // 打开视频文件
//     cv::VideoCapture cap("E:\\code\\YOLO\\yolov5-6.0_cpp\\output_video.avi"); // 请替换为你的输入视频路径

//     // 检查视频文件是否成功打开
//     if (!cap.isOpened()) {
//         std::cerr << "无法打开视频文件!" << std::endl;
//         return -1;
//     }

//     // 获取视频的帧率、宽度和高度
//     double fps = cap.get(cv::CAP_PROP_FPS);  // 帧率
//     int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
//     int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
//     int frameCount = cap.get(cv::CAP_PROP_FRAME_COUNT);

//     // 设置输出视频
//     cv::VideoWriter writer;
//     writer.open("output_video_normal.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps * 12, cv::Size(width, height));

//     // 检查 VideoWriter 是否成功打开
//     if (!writer.isOpened()) {
//         std::cerr << "无法打开视频写入器!" << std::endl;
//         return -1;
//     }

//     cv::Mat frame;
//     int frameIndex = 0;

//     // 以2倍速跳过帧并保存
//     while (cap.read(frame)) {
//         if (frameIndex % 1 == 0) { // 每隔1帧处理一帧，跳过一帧
//             writer.write(frame);
//         }
//         frameIndex++;
//     }

//     // 释放资源
//     cap.release();
//     writer.release();

//     std::cout << "视频处理完成，输出保存在 output_video_normal.avi" << std::endl;

//     return 0;
// }
