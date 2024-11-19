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
#include <nav_msgs/msg/odometry.hpp>
#include <rover_msgs/msg/object_detections.hpp>
#include <rover_msgs/msg/object_detection.hpp>

//Testing service to start detections
#include <std_srvs/srv/set_bool.hpp>


using namespace nvinfer1;
#define NMS_THRESH 0.4  //Not sure where this is used

#ifndef DEG2RAD
#define DEG2RAD 0.017453293
#define RAD2DEG 57.295777937
#endif

// Basic structure to compare timestamps of a sensor. Determines if a specific sensor data has been updated or not.
struct TimestampHandler {

    // Compare the new timestamp to the last valid one. If it is higher, save it as new reference.
    inline bool isNew(sl::Timestamp& ts_curr, sl::Timestamp& ts_ref) {
        bool new_ = ts_curr > ts_ref;
        if (new_) ts_ref = ts_curr;
        return new_;
    }
    // Specific function for IMUData.
    inline bool isNew(sl::SensorsData::IMUData& imu_data) {
        return isNew(imu_data.timestamp, ts_imu);
    }
    // Specific function for MagnetometerData.
    inline bool isNew(sl::SensorsData::MagnetometerData& mag_data) {
        return isNew(mag_data.timestamp, ts_mag);
    }
    // Specific function for BarometerData.
    inline bool isNew(sl::SensorsData::BarometerData& baro_data) {
        return isNew(baro_data.timestamp, ts_baro);
    }

    sl::Timestamp ts_imu = 0, ts_baro = 0, ts_mag = 0; // Initial values
};

// Function to display sensor parameters.
void printSensorConfiguration(sl::SensorParameters& sensor_parameters) {
    if (sensor_parameters.isAvailable) {
        std::cout << "*****************************" << std::endl;
        std::cout << "Sensor Type: " << sensor_parameters.type << std::endl;
        // std::cout << "Max Rate: "    << sensor_parameters.sampling_rate << SENSORS_UNIT::HERTZ << std::endl;
        std::cout << "Max Rate: "    << sensor_parameters.sampling_rate << std::endl;
        std::cout << "Range: ["      << sensor_parameters.range << "] " << sensor_parameters.sensor_unit << std::endl;
        std::cout << "Resolution: "  << sensor_parameters.resolution << " " << sensor_parameters.sensor_unit << std::endl;
        if (isfinite(sensor_parameters.noise_density)) std::cout << "Noise Density: " << sensor_parameters.noise_density <<" "<< sensor_parameters.sensor_unit<<"/√Hz"<<std::endl;
        if (isfinite(sensor_parameters.random_walk)) std::cout << "Random Walk: " << sensor_parameters.random_walk <<" "<< sensor_parameters.sensor_unit<<"/s/√Hz"<<std::endl;
    }
} 


class ObjectDetectionNode : public rclcpp::Node {
public:
    ObjectDetectionNode() : Node("object_detection") {
        // Publishers
        image_transport::ImageTransport it(this->shared_from_this());
        detection_annotation_ = it.advertise("/object_detection/annotated", 10);

        object_detection_pub_ = this->create_publisher<rover_msgs::msg::ObjectDetections>("/object_detection", 10);
        
        // Declare the 'engine_name' parameter with an empty string as the default value
        this->declare_parameter<std::string>("engine_name", "");
        std::string engine_name;
        this->get_parameter("engine_name", engine_name);
        // Period in milliseconds for frame process 
        int process_period_ms;      
        this->declare_parameter<int>("process_period_ms", 2000);
        this->get_parameter("process_period_ms", process_period_ms);
        // Period in milliseconds for sensor data
        int sensor_process_period_ms;      
        this->declare_parameter<int>("process_period_ms", 20);
        this->get_parameter("sensor_process_period_ms", sensor_process_period_ms);

        this->declare_parameter<float>("confidence_thresh", 0.3);
        this->get_parameter("confidence_thresh", this->conf_thresh);

        // Initialize pose with identity
        cam_w_pose.pose_data.setIdentity();

        setup_node();

        setup_yolo(engine_name);  
        // Timer for the detection loop
        timer_ = this->create_wall_timer(std::chrono::milliseconds(process_period_ms), std::bind(&ObjectDetectionNode::processFrame, this));


        /* SETUP IMU, MAG, ODOM Publishers */
        imu_publisher_ = this->create_publisher<sensor_msgs::msg::Imu>("zed/imu/data", 10);
        mag_publisher_ = this->create_publisher<sensor_msgs::msg::MagneticField>("zed/imu/mag", 10);
        odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("zed/zed_node/odom", 10);

        //TODO: TEST this timer to run seperately from the zed obstacles
        timer_sensors_ = this->create_wall_timer(
            std::chrono::milliseconds(sensor_process_period_ms), std::bind(&ObjectDetectionNode::process_ZED_data, this));

        //Testing service call to start the object detection
        service_ = this->create_service<std_srvs::srv::SetBool>(
            "toggle_object_detection",
            std::bind(&ObjectDetectionNode::handleToggleDetect, this, std::placeholders::_1, std::placeholders::_2)
        );
        
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
    
    void handleToggleDetect(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                                std::shared_ptr<std_srvs::srv::SetBool::Response> response){
        run_detector_ = request->data;
        response->success = true;
        response->message = run_detector_ ? "Detection enabled" : "Detection disabled";
        RCLCPP_INFO(this->get_logger(), "Toggled Detection: %s", response->message.c_str());
    }

    void processFrame() {
        if (!run_detector_) {
            return;  // Do nothing if detections is disabled
        }
        // Grab image from ZED and process detections
        if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
            /* Left image */
            // zed.retrieveImage(left_sl, sl::VIEW::LEFT);
            /* Object detections */
            publishDetections();

            // publish_sensor_data();
        }
    }

    void process_ZED_data(){
        if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
            publish_sensor_data();
        }
    }

    void publish_sensor_data(){
        sl::SensorsData sensors_data;
        sl::Pose camera_pose;

        zed.getSensorsData(sensors_data, sl::TIME_REFERENCE::CURRENT);
        zed.getPosition(camera_pose, sl::REFERENCE_FRAME::WORLD);

        if (ts.isNew(sensors_data.imu)) {
            std::cout << "IMU Orientation: {" << sensors_data.imu.pose.getOrientation() << "}" << std::endl;
            std::cout << "IMU Linear Acceleration: {" << sensors_data.imu.linear_acceleration << "} [m/sec^2]" << std::endl;
            std::cout << "IMU Angular Velocity: {" << sensors_data.imu.angular_velocity << "} [deg/sec]" << std::endl;
        }

        // Check if Magnetometer data has been updated
        // if (ts.isNew(sensors_data.magnetometer)) {
        std::cout << " Magnetometer Magnetic Field: {" << sensors_data.magnetometer.magnetic_field_calibrated << "} [uT]" << std::endl;
        // }

        auto current_timestamp = this->now();

        // Publish IMU data
        auto imu_msg = std::make_unique<sensor_msgs::msg::Imu>();
        imu_msg->header.stamp = current_timestamp;
        imu_msg->header.frame_id = "zed_imu_link";

        auto imu_data = sensors_data.imu;
        auto mag_data = sensors_data.magnetometer;
        imu_msg->orientation.x = imu_data.pose.getOrientation().x;
        imu_msg->orientation.y = imu_data.pose.getOrientation().y;
        imu_msg->orientation.z = imu_data.pose.getOrientation().z;
        imu_msg->orientation.w = imu_data.pose.getOrientation().w;
        imu_msg->angular_velocity.x = imu_data.angular_velocity.x;
        imu_msg->angular_velocity.y = imu_data.angular_velocity.y;
        imu_msg->angular_velocity.z = imu_data.angular_velocity.z;
        imu_msg->linear_acceleration.x = imu_data.linear_acceleration.x;
        imu_msg->linear_acceleration.y = imu_data.linear_acceleration.y;
        imu_msg->linear_acceleration.z = imu_data.linear_acceleration.z;

        imu_publisher_->publish(std::move(imu_msg));

        // Publish magnetometer data
        auto mag_msg = std::make_unique<sensor_msgs::msg::MagneticField>();
        mag_msg->header.stamp = current_timestamp;
        mag_msg->header.frame_id = "zed_imu_link";
        mag_msg->magnetic_field.x = mag_data.magnetic_field_calibrated.x;
        mag_msg->magnetic_field.y = mag_data.magnetic_field_calibrated.y;
        mag_msg->magnetic_field.z = mag_data.magnetic_field_calibrated.z;

        mag_publisher_->publish(std::move(mag_msg));

        // Publish odometry data
        auto odom_msg = std::make_unique<nav_msgs::msg::Odometry>();
        odom_msg->header.stamp = current_timestamp;
        odom_msg->header.frame_id = "odom";
        odom_msg->child_frame_id = "zed_base_link";

        odom_msg->pose.pose.position.x = camera_pose.getTranslation().x;
        odom_msg->pose.pose.position.y = camera_pose.getTranslation().y;
        odom_msg->pose.pose.position.z = camera_pose.getTranslation().z;
        odom_msg->pose.pose.orientation.x = camera_pose.getOrientation().x;
        odom_msg->pose.pose.orientation.y = camera_pose.getOrientation().y;
        odom_msg->pose.pose.orientation.z = camera_pose.getOrientation().z;
        odom_msg->pose.pose.orientation.w = camera_pose.getOrientation().w;

        // odom_msg->twist.twist.linear.x = camera_pose.getVelocity().x;
        // odom_msg->twist.twist.linear.y = camera_pose.getVelocity().y;
        // odom_msg->twist.twist.linear.z = camera_pose.getVelocity().z;
        odom_msg->twist.twist.angular.x = camera_pose.getEulerAngles().x;
        odom_msg->twist.twist.angular.y = camera_pose.getEulerAngles().y;
        odom_msg->twist.twist.angular.z = camera_pose.getEulerAngles().z;

        odom_publisher_->publish(std::move(odom_msg));
    }

    void publishDetections() {
        // Running inference
        //CHECK TO MAKE SURE: THIS MIGHT NEED TO RUN ONLY WHEN WE WANT IT TO hence the subscriber count
        zed.retrieveImage(left_sl, sl::VIEW::LEFT);
   
        auto detections = detector_.run(left_sl, display_resolution_.height, display_resolution_.width, this->conf_thresh);
        
        // Convert the sl::Mat to cv::Mat
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
        // if (detection_annotation_.getNumSubscribers() > 0) {
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
        // }


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
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr timer_sensors_;

    float conf_thresh;

    TimestampHandler ts;  //TODO: get the time stamp handler to only publish new data

    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::MagneticField>::SharedPtr mag_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    //TODO: Test second timer to run timing of objects and data independently
    // rclcpp::TimerBase::SharedPtr timer_;


    //IMAGE TRANSPORT
    image_transport::Publisher detection_annotation_;
    
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr service_;
    bool run_detector_ = false; //Flag to run object detection
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
