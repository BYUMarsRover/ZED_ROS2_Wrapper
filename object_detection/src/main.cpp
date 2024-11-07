#include <iostream>
#include <chrono>
#include <cmath>
#include <memory>
#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"

#include "yolo.hpp"
// #include "sl_tools.h"

#include <sl/Camera.hpp>
#include <NvInfer.h>
#include <cv_bridge/cv_bridge.h>

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/magnetic_field.hpp>
#include <rover_msgs/msg/object_detections.hpp>
#include <rover_msgs/msg/object_detection.hpp>



using namespace nvinfer1;
#define NMS_THRESH 0.4  //Not sure where this is used

#ifndef DEG2RAD
#define DEG2RAD 0.017453293
#define RAD2DEG 57.295777937
#endif


class ObjectDetectionNode : public rclcpp::Node {
public:
    ObjectDetectionNode() : Node("object_detection") {
        // Publishers
        // rclcpp::Node::SharedPtr node = rclcpp::Node::make_shared("image_publisher");
        // image_transport::ImageTransport it(node);
        // image_transport::Publisher detection_annotation_ = it.advertise("/object_detection/annotated", 1);
        // detection_annotation_ = image_transport::ImageTransport(this).advertise("/object_detection/annotated", 1);
        object_detection_pub_ = this->create_publisher<rover_msgs::msg::ObjectDetections>("/object_detection", 10);
        
        // Declare the 'engine_name' parameter with an empty string as the default value
        this->declare_parameter<std::string>("engine_name", "");
        std::string engine_name;
        this->get_parameter("engine_name", engine_name);
        // Period in milliseconds for frame process 
        int process_period_ms;      
        this->declare_parameter<int>("process_period_ms", 2000);
        this->get_parameter("process_period_ms", process_period_ms);

        this->declare_parameter<float>("confidence_thresh", 0.3);
        this->get_parameter("confidence_thresh", this->conf_thresh);

        // Initialize pose with identity
        cam_w_pose.pose_data.setIdentity();

        setup_node();

        setup_yolo(engine_name);  
        // Timer for the detection loop
        timer_ = this->create_wall_timer(std::chrono::milliseconds(process_period_ms), std::bind(&ObjectDetectionNode::processFrame, this));
    }

private:

    void setup_yolo(std::string engine_name){
        /* Custom YOLOv8 model initialization */
        if (!engine_name.empty()) {
            std::cout << "Using YOLOv8 model engine: " << engine_name.c_str() << std::endl;
        } else {
            std::cout << "No YOLOv8 model engine specified." << std::endl;
            throw std::runtime_error("Engine not specified.");
            return;
        }
        if (detector_.init(engine_name)) {
            std::cout << "Detector init failed!" << std::endl;
            throw std::runtime_error("Detector init failed");
            return;
        }
        std::cout << "Initialized object detection" << std::endl;
    }

    cv::Rect get_rect(BBox box) {
        return cv::Rect(round(box.x1), round(box.y1), round(box.x2 - box.x1), round(box.y2 - box.y1));
    }

    std::vector<sl::uint2> cvt(const BBox &bbox_in) {
        std::vector<sl::uint2> bbox_out(4);
        bbox_out[0] = sl::uint2(bbox_in.x1, bbox_in.y1);
        bbox_out[1] = sl::uint2(bbox_in.x2, bbox_in.y1);
        bbox_out[2] = sl::uint2(bbox_in.x2, bbox_in.y2);
        bbox_out[3] = sl::uint2(bbox_in.x1, bbox_in.y2);
        return bbox_out;
    }

       
    void setup_node(){
        //TODO: Check to make sure below is not used anymore
        // sensor_msgs::CameraInfoPtr left_camera_info_msg;
        // left_camera_info_msg.reset(new sensor_msgs::CameraInfo());
        // std::string left_camera_frame_id = "zed2i_left_camera_optical_frame";
        

        /* ZED camera initializaion */
        // Opening the ZED camera before the model deserialization to avoid cuda context issue
        
        init_parameters.sdk_verbose = true;
        init_parameters.input.setFromSerialNumber(20382332);
        init_parameters.depth_mode = sl::DEPTH_MODE::ULTRA;

        std::cout << std::to_string(zed.getCameraInformation().serial_number) << std::endl;

        // Open the camera
        auto returned_state = zed.open(init_parameters);
        if (returned_state != sl::ERROR_CODE::SUCCESS) {
            RCLCPP_ERROR(this->get_logger(), "Camera Open failed with error code: %s. Exit program.", sl::toString(returned_state).c_str());
            throw std::runtime_error("Camera initialization failed.");
            return;
        }
        zed.enablePositionalTracking();

        /* ZED object detection initialization */
        sl::ObjectDetectionParameters detection_parameters;
        detection_parameters.enable_tracking = true;
        //detection_parameters.enable_segmentation = false; // designed to give person pixel mask, ZED SDK 4 only
        detection_parameters.detection_model = sl::OBJECT_DETECTION_MODEL::CUSTOM_BOX_OBJECTS;
        returned_state = zed.enableObjectDetection(detection_parameters);
        if (returned_state != sl::ERROR_CODE::SUCCESS) {
            RCLCPP_ERROR(this->get_logger(), "enableObjectDetection failed with error code: %s. Exit program.", sl::toString(returned_state).c_str());
            zed.close();
            throw std::runtime_error("Object detection enable failed.");
            return;
        }


        auto camera_config = zed.getCameraInformation().camera_configuration;
        sl::Resolution pc_resolution(std::min((int) camera_config.resolution.width, 720), std::min((int) camera_config.resolution.height, 404));
        auto camera_info = zed.getCameraInformation(pc_resolution).camera_configuration;

        std::cout << "Initialized ZED camera" << std::endl;

        /* Object detection data initialization */
        display_resolution_ = zed.getCameraInformation().camera_configuration.resolution;
        
    }
    
    
    void processFrame() {
        // Grab image from ZED and process detections
        if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
            /* Left image */
            // zed.retrieveImage(left_sl, sl::VIEW::LEFT);
            /* Object detections */
            publishDetections();
        }
    }

    void publishDetections() {
        // Running inference
        //CHECK TO MAKE SURE: THIS MIGHT NEED TO RUN ONLY WHEN WE WANT IT TO hence the subscriber count
        zed.retrieveImage(left_sl, sl::VIEW::LEFT);
        // Convert the sl::Mat to cv::Mat
   
        auto detections = detector_.run(left_sl, display_resolution_.height, display_resolution_.width, this->conf_thresh);
        
        left_cv_ = slMat2cvMat(left_sl);

        // Preparing for ZED SDK ingesting
        std::vector<sl::CustomBoxObjectData> objects_in;
        
        for (auto &it : detections) {
            sl::CustomBoxObjectData tmp;
            // Fill the detections into the correct format
            tmp.unique_object_id = sl::generate_unique_id();
            tmp.probability = it.prob;
            tmp.label = (int) it.label;
            tmp.bounding_box_2d = cvt(it.box);
            tmp.is_grounded = ((int) it.label == 0); // Only the first class (person) is grounded, that is moving on the floor plane
            // others are tracked in full 3D space
            objects_in.push_back(tmp);
        }
        // Send the custom detected boxes to the ZED
        zed.ingestCustomBoxObjects(objects_in);


        //DETECTION ANNOTATION PUBLISHER HERE


        // Publish detections if subscribers are present
        // if (object_detection_pub_->get_subscription_count() > 0) {
        zed.retrieveObjects(objects, object_tracker_params_rt_);
        // std::cout << "Object Detection Check" << std::endl;
        
        if (!objects.object_list.empty()) {
            rover_msgs::msg::ObjectDetections msg;
            msg.header.stamp = now();
            msg.header.frame_id = "object_detections";
            
            for (const auto& object : objects.object_list) {
                rover_msgs::msg::ObjectDetection detection;
                detection.id = object.id;
                detection.label = object.raw_label;
                detection.x = object.position.x;
                detection.y = object.position.y;
                detection.z = object.position.z;
                detection.confidence = object.confidence;
                msg.objects.push_back(detection);
                // std::cout << "Object Detected" << std::endl;
            }

            object_detection_pub_->publish(msg);
        }
        // }

        // TODO: WE MIGHT NEED THE CVT FUNCTION IN HERE FOR ANNotations??
        // THE OLD CODE LOOKS A LITTLE DIFFERENT BELOW THIS
        // Annotate and publish image
        if (detection_annotation_.getNumSubscribers() > 0) {
            left_cv_ = slMat2cvMat(left_sl);
            for (const auto& detection : detections) {
                cv::Rect r = get_rect(detection.box);
                cv::rectangle(left_cv_, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(left_cv_, std::to_string(static_cast<int>(detection.label)), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
            
            std_msgs::msg::Header header;
            header.stamp = this->now(); // Set the current time
            header.frame_id = "object_detection"; // Set appropriate frame ID
            cv_bridge::CvImage annotated_img(header, sensor_msgs::image_encodings::TYPE_8UC4, left_cv_);
            detection_annotation_.publish(annotated_img.toImageMsg());
        }


    }

    // Member variables
    sl::Camera zed;
    Yolo detector_;
    cv::Mat left_cv_;
    sl::Objects objects;
    sl::ObjectDetectionRuntimeParameters object_tracker_params_rt_;
    sl::Resolution display_resolution_;
    sl::InitParameters init_parameters;

    /* Resolution calcualations */
    sl::Resolution resolution;


    sl::Mat left_sl, point_cloud;
    sl::Pose cam_w_pose = sl::Pose();

    // std::string mag_frame_id = "zed2i_mag_link";
    // std::string imu_frame_id = "zed2i_imu_link";    

    rclcpp::Publisher<rover_msgs::msg::ObjectDetections>::SharedPtr object_detection_pub_;
    image_transport::Publisher detection_annotation_;
    rclcpp::TimerBase::SharedPtr timer_;

    float conf_thresh;
};


int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    try {
        rclcpp::spin(std::make_shared<ObjectDetectionNode>());
    } catch (const std::runtime_error &e) {
        RCLCPP_FATAL(rclcpp::get_logger("rclcpp"), "Node terminated due to error: %s", e.what());
    }
    rclcpp::shutdown();
    return 0;
}
