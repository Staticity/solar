#include <iostream>
#include <string>
#include <vector>
#include "SpiceUsr.h"

#include <sophus/se3.hpp>

class SolarSystemPose {
 public:
  SolarSystemPose() { init(); }
  ~SolarSystemPose() { cleanup(); }

  Sophus::SE3d T_J2000_body(const std::string& body, double ephemerisTime) {
    SpiceDouble moonState[6], lt_moon;
    spkezr_c(
        body.c_str(),
        ephemerisTime,
        "J2000",
        "NONE",
        "EARTH",
        moonState,
        &lt_moon);

    Sophus::SE3d estT_J2000_body;
    estT_J2000_body.translation() =
        Eigen::Vector3d(moonState[0], moonState[1], moonState[2]);

    const std::string& fixedFrame = "IAU_" + body;
    SpiceDouble rotationMatrix[3][3];
    pxform_c("J2000", fixedFrame.c_str(), ephemerisTime, rotationMatrix);
    SpiceDouble quaternion[4];
    m2q_c(rotationMatrix, quaternion);
    estT_J2000_body.setQuaternion(Eigen::Quaterniond(
        quaternion[0], quaternion[1], quaternion[2], quaternion[3]));

    return estT_J2000_body;
  }

 private:
  const std::vector<std::string> kernelPaths_ = {
      "/Users/static/Downloads/cspice/data/naif0012.tls",
      "/Users/static/Downloads/cspice/data/de430.bsp",
      "/Users/static/Downloads/cspice/data/pck00011.tpc"};

  void init() {
    for (const auto& path : kernelPaths_) {
      furnsh_c(path.c_str());
    }
  }

  void cleanup() {
    for (const auto& path : kernelPaths_) {
      unload_c(path.c_str());
    }
  }
};

int main() {
  SolarSystemPose solar;

  // Define the UTC time for which you want the data
  ConstSpiceChar* utc = "2021-09-30T12:00:00";

  // Convert the UTC time to ephemeris time (TDB)
  SpiceDouble et;
  str2et_c(utc, &et);

  std::vector<std::string> bodies = {"SUN", "EARTH", "MOON"};
  for (const auto& body : bodies) {
    const Sophus::SE3d T_J2000_object = solar.T_J2000_body(body, et);
    std::cout << body << std::endl;
    std::cout << T_J2000_object.matrix() << std::endl;
  }
}