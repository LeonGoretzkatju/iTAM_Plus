// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

// Note: do not modify Open3D.h, modify Open3D.h.in instead
#include "Open3D/Camera/PinholeCameraIntrinsic.h"
#include "Open3D/Camera/PinholeCameraParameters.h"
#include "Open3D/Camera/PinholeCameraTrajectory.h"
#include "Open3D/ColorMap/ColorMapOptimization.h"
#include "Open3D/ColorMap/ImageWarpingField.h"
#include "Open3D/GUI/Application.h"
#include "Open3D/GUI/Button.h"
#include "Open3D/GUI/Checkbox.h"
#include "Open3D/GUI/Color.h"
#include "Open3D/GUI/Combobox.h"
#include "Open3D/GUI/Dialog.h"
#include "Open3D/GUI/Gui.h"
#include "Open3D/GUI/ImageLabel.h"
#include "Open3D/GUI/Label.h"
#include "Open3D/GUI/Layout.h"
#include "Open3D/GUI/Menu.h"
#include "Open3D/GUI/ProgressBar.h"
#include "Open3D/GUI/SceneWidget.h"
#include "Open3D/GUI/Slider.h"
#include "Open3D/GUI/TabControl.h"
#include "Open3D/GUI/TextEdit.h"
#include "Open3D/GUI/Theme.h"
#include "Open3D/GUI/Window.h"
#include "Open3D/Geometry/BoundingVolume.h"
#include "Open3D/Geometry/Geometry.h"
#include "Open3D/Geometry/HalfEdgeTriangleMesh.h"
#include "Open3D/Geometry/Image.h"
#include "Open3D/Geometry/KDTreeFlann.h"
#include "Open3D/Geometry/LineSet.h"
#include "Open3D/Geometry/Octree.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/RGBDImage.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Geometry/VoxelGrid.h"
#include "Open3D/IO/ClassIO/FeatureIO.h"
#include "Open3D/IO/ClassIO/FileFormatIO.h"
#include "Open3D/IO/ClassIO/IJsonConvertibleIO.h"
#include "Open3D/IO/ClassIO/ImageIO.h"
#include "Open3D/IO/ClassIO/LineSetIO.h"
#include "Open3D/IO/ClassIO/PinholeCameraTrajectoryIO.h"
#include "Open3D/IO/ClassIO/PointCloudIO.h"
#include "Open3D/IO/ClassIO/PoseGraphIO.h"
#include "Open3D/IO/ClassIO/TriangleMeshIO.h"
#include "Open3D/IO/ClassIO/VoxelGridIO.h"
#include "Open3D/Integration/ScalableTSDFVolume.h"
#include "Open3D/Integration/TSDFVolume.h"
#include "Open3D/Integration/UniformTSDFVolume.h"
#include "Open3D/Odometry/Odometry.h"
#include "Open3D/Open3DConfig.h"
#include "Open3D/Registration/Feature.h"
#include "Open3D/Registration/Registration.h"
#include "Open3D/Registration/TransformationEstimation.h"
#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/Eigen.h"
#include "Open3D/Utility/FileSystem.h"
#include "Open3D/Utility/Helper.h"
#include "Open3D/Utility/Timer.h"
#include "Open3D/Visualization/Utility/DrawGeometry.h"
#include "Open3D/Visualization/Utility/SelectionPolygon.h"
#include "Open3D/Visualization/Utility/SelectionPolygonVolume.h"
#include "Open3D/Visualization/Visualizer/ViewControl.h"
#include "Open3D/Visualization/Visualizer/ViewControlWithCustomAnimation.h"
#include "Open3D/Visualization/Visualizer/ViewControlWithEditing.h"
#include "Open3D/Visualization/Visualizer/Visualizer.h"
#include "Open3D/Visualization/Visualizer/VisualizerWithCustomAnimation.h"
#include "Open3D/Visualization/Visualizer/VisualizerWithEditing.h"
#include "Open3D/Visualization/Visualizer/VisualizerWithKeyCallback.h"
#include "Open3D/Visualization/Visualizer/VisualizerWithVertexSelection.h"

// clang-format off
//#include "Open3D/IO/Sensor/AzureKinect/AzureKinectRecorder.h"
//#include "Open3D/IO/Sensor/AzureKinect/AzureKinectSensorConfig.h"
//#include "Open3D/IO/Sensor/AzureKinect/AzureKinectSensor.h"
//#include "Open3D/IO/Sensor/AzureKinect/MKVMetadata.h"
//#include "Open3D/IO/Sensor/AzureKinect/MKVReader.h"
//#include "Open3D/IO/Sensor/AzureKinect/MKVWriter.h"
//#include "Open3D/IO/Sensor/RGBDRecorder.h"
//#include "Open3D/IO/Sensor/RGBDSensorConfig.h"
//#include "Open3D/IO/Sensor/RGBDSensor.h"
