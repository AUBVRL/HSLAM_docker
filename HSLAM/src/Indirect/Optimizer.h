#ifndef __OPTIMIZER_H_
#define __OPTIMIZER_H_

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/io_helper.h>
#include "FullSystem/HessianBlocks.h"
#include "util/NumType.h"

#include <g2o/types/sba/types_six_dof_expmap.h>

namespace g2o{
    struct Sim3;
}

namespace HSLAM
{
    class Frame;
    class Map;
    class MapPoint;
    class FullSystem;
    class Sim3Vertex;

    namespace OptimizationStructs
    {
        using namespace g2o;
        using namespace Eigen;

        inline Vec3 invert_depth(const Vec3 &x)
            {
                g2o::Vector3 res;
                res(0) = x(0);
                res(1) = x(1);
                res(2) = 1;
                return res / x(2);
            }

        class vertexSE3 : public BaseVertex<6, SE3>
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            vertexSE3() : BaseVertex<6, SE3>() { }
            bool read(std::istream &is) override { return true; }
            bool write(std::ostream &os) const override { return true; }

            virtual void setToOriginImpl() override {_estimate = SE3();}
            virtual void oplusImpl(const double *update_) override
            {
                Eigen::Map<const Vec6> update(update_);
                _estimate = SE3::exp(update) * _estimate;
            }
        };

        class edgeSE3XYZPoseOnly : public BaseUnaryEdge<2, Vec2, vertexSE3>
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
            edgeSE3XYZPoseOnly() : BaseUnaryEdge<2, Vec2, vertexSE3>() { scale = 1.0; }

            bool read(std::istream &is) override { return true; }
            bool write(std::ostream &os) const override { return true; }

            void computeError() override
            {
                const vertexSE3 *v1 = static_cast<vertexSE3 *>(_vertices[0]);
                

                Vec3 PointinFrame = v1->estimate() * (Xw);
                PointinFrame = PointinFrame * (1.0 / PointinFrame[2]);
                if (PointinFrame[2] < 0) 
                {
                    std::cout << "negative projected depth should not occur! skip"<<std::endl;
                    return;
                }
                double u = fx * PointinFrame[0] + cx;
                double v = fy * PointinFrame[1] + cy;
                _error =  (_measurement - Vec2(u,v))/scale; //+ Vec2(0.5,0.5)
                return;
            }

            void setScale(double _scale)
            {
                scale = (double)_scale;
            }

            void setCamera(const double _fx, const double _fy, const double _cx, const double _cy) { fx = (double)_fx; fy = (double)_fy; cx = (double)_cx; cy = (double)_cy; }

            void setXYZ(const Vector3d &_Xw) {Xw = _Xw;}

            bool isDepthPositive()
            {
                const vertexSE3 *v1 = static_cast<const vertexSE3 *>(_vertices[0]);
                Vec3 PointinFrame = v1->estimate() * (Xw);
                PointinFrame = PointinFrame * (1.0 / PointinFrame[2]);
                return (PointinFrame[2] > 0);
            }
            void linearizeOplus() override
            {
                vertexSE3 *vi = static_cast<vertexSE3 *>(_vertices[0]);
                Vec3 xyz_trans = vi->estimate() * (Xw);

                double x = xyz_trans[0];
                double y = xyz_trans[1];
                double invz = 1.0 / xyz_trans[2];
                double invz_2 = invz * invz;

                _jacobianOplusXi(0, 0) = -invz * fx; 
                _jacobianOplusXi(0, 1) = 0;
                _jacobianOplusXi(0, 2) = x * invz_2 * fx;
                _jacobianOplusXi(0, 3) = x * y * invz_2 * fx;
                _jacobianOplusXi(0, 4) = -(1 + (x * x * invz_2)) * fx;
                _jacobianOplusXi(0, 5) = y * invz * fx;
        
                _jacobianOplusXi(1, 0) = 0;
                _jacobianOplusXi(1, 1) = -invz * fy;
                _jacobianOplusXi(1, 2) = y * invz_2 * fy;
                _jacobianOplusXi(1, 3) = (1 + y * y * invz_2) * fy;
                _jacobianOplusXi(1, 4) = -x * y * invz_2 * fy;
                _jacobianOplusXi(1, 5) = -x * invz * fy;
               
            }
            Vec3 Xw;
        private:
           
            double fx, fy, cx, cy, scale;
        };

        class camParams : public g2o::Parameter
        { //for some reason the default cameraParameters use a single focal length fx! rewrite necessary classes for fx fy
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
            
            camParams(double fx, double fy, double cx, double cy): focal_length(focal_length), principle_point(principle_point)
            {
                focal_length[0] = fx;
                focal_length[1] = fy;
                principle_point[0] = cx;
                principle_point[1] = cx;
            }
            Vec2 cam_map(const g2o::Vector3 &trans_xyz) const
            {
                Vec2 proj;
                proj[0] = trans_xyz[0] / trans_xyz[2];
                proj[1] = trans_xyz[1] / trans_xyz[2];
                Vec2 res;
                res[0] = proj[0]*focal_length[0] + principle_point[0];
                res[1] = proj[1]*focal_length[1] + principle_point[1];
                return res;
            }

            Vec3 camInvMap(const Vec2 & coords) const
            {
                Vec3 Out;
                Out[0] = (1.0 / focal_length[0]) * (coords[0] - principle_point[0]);
                Out[1] = (1.0 / focal_length[1]) * (coords[1] - principle_point[1]);
                Out[2] = 1.0;
                
                return Out;
            }


            virtual bool read(std::istream &is)
            {
                is >> focal_length[0];
                is >> focal_length[1];
                is >> principle_point[0];
                is >> principle_point[1];
                return true;
            }

            virtual bool write(std::ostream &os) const
            {
                os << focal_length[0] << " ";
                os << focal_length[1] << " ";
                os << principle_point[0] << " ";
                os << principle_point[1] << " ";
                return true;
            }

            Vec2 focal_length;
            Vec2 principle_point;
        };

        class VertexPointDepth : public BaseVertex<1, double>
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            VertexPointDepth(){}
            bool read(std::istream &is)
            {
                is >> UV[0] ;
                is >> UV[1];
                is >> _estimate;
                return is.good() || is.eof();
            }
            bool write(std::ostream &os) const
            {
                os << UV[0] << " " << UV[1] << " " << estimate();
                return os.good();
            }

            virtual void setToOriginImpl()
            {
                _estimate = 0;
            }

            virtual void oplusImpl(const number_t *update)
            {
                _estimate += *update;
            }

            void setUV(Vec2 _UV)
            {
                UV = _UV;
            }

            Vec2 UV;
        };

        class EdgeProjectinvDepth : public g2o::BaseMultiEdge<2, Vec2>
        {
        public:
        //the following parameterization stores the map points in VertexSBAPointXYZ struct
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            EdgeProjectinvDepth()
            {
                resize(3); //set the number of vertices
                resizeParameters(1); //set the number of parameters to a class (camParams) that contains all params
                installParameter(_cam, 0, 0); //register the camparams class must be defined in the function creating the edges
            }

            bool read(std::istream &is)
            {
                readParamIds(is);
                g2o::internal::readVector(is, _measurement);
                return readInformationMatrix(is);
            }
            bool write(std::ostream &os) const
            {
                writeParamIds(os);
                g2o::internal::writeVector(os, measurement());
                return writeInformationMatrix(os);
            }

            void computeError()
            {
                const VertexPointDepth *psi = static_cast<const VertexPointDepth *>(_vertices[0]);
                const VertexSE3Expmap *T_p_from_world = static_cast<const VertexSE3Expmap *>(_vertices[1]);
                const VertexSE3Expmap *T_anchor_from_world = static_cast<const VertexSE3Expmap *>(_vertices[2]);
                const camParams *cam = static_cast<const camParams *>(parameter(0));

                g2o::Vector2 obs(_measurement);
                // _error = obs - cam->cam_map(T_p_from_world->estimate() * T_anchor_from_world->estimate().inverse() * invert_depth(psi->estimate()));
                _error = obs - cam->cam_map(T_p_from_world->estimate() * T_anchor_from_world->estimate().inverse() * invert_depth( cam->camInvMap(psi->UV) * (1.0/ psi->estimate()) ));
                
            }

            Matrix<double, 2, 3, Eigen::ColMajor> d_proj_d_y(const Vec2 &f, const Vec3 &xyz)
            {
                double z_sq = xyz[2] * xyz[2];
                Matrix<double, 2, 3, Eigen::ColMajor> J;
                J << f[0] / xyz[2], 0, -(f[0] * xyz[0]) / z_sq,
                    0, f[1] / xyz[2], -(f[1] * xyz[1]) / z_sq;
                return J;
            }

            Matrix<double, 3, 6, Eigen::ColMajor> d_expy_d_y(const Vec3 &y)
            {
                Matrix<double, 3, 6, Eigen::ColMajor> J;
                J.topLeftCorner<3, 3>() = -skew(y);
                J.bottomRightCorner<3, 3>().setIdentity();
                return J;
            }

            inline Mat33 d_Tinvpsi_d_psi(const SE3Quat &T, const Vec3 &psi)
            {
                Mat33 R = T.rotation().toRotationMatrix();
                Vec3 x = invert_depth(psi);
                Vec3 r1 = R.col(0);
                Vec3 r2 = R.col(1);
                Mat33 J;
                J.col(0) = r1;
                J.col(1) = r2;
                J.col(2) = -R * x;
                J *= 1. / psi.z();
                return J;
            }

            // void linearizeOplus()
            // {
            //     VertexPointDepth *vpoint = static_cast<VertexPointDepth *>(_vertices[0]);
            //     double psi_a = vpoint->estimate();
            //     VertexSE3Expmap *vpose = static_cast<VertexSE3Expmap *>(_vertices[1]);
            //     SE3Quat T_cw = vpose->estimate();
            //     VertexSE3Expmap *vanchor = static_cast<VertexSE3Expmap *>(_vertices[2]);
            //     const camParams *cam = static_cast<const camParams *>(parameter(0));
            //     SE3Quat A_aw = vanchor->estimate();
            //     SE3Quat T_ca = T_cw * A_aw.inverse();

            //     Vec3 x1 = cam->camInvMap(vpoint->UV) * (1.0 / vpoint->estimate());
            //     Vector3 x_a = invert_depth(x1);
            //     Vector3 y = T_ca * x_a;

            //     Matrix<double, 2, 3, Eigen::ColMajor> Jcam = d_proj_d_y(cam->focal_length, y);
            //     Matrix<double, 3, 3> d_invert_depth;
            //     double z2 = x1(2) * x1(2);
            //     d_invert_depth << 1.0 / x1(2), 0.0, -x1(0) / z2, 0.0, 1.0 / x1(2), -x1(1) / z2, 0.0, 0.0, -1.0 / z2;
            //     _jacobianOplus[0] = -Jcam * T_ca.rotation().toRotationMatrix() * d_invert_depth * (x1 /vpoint->estimate());
            //     _jacobianOplus[1] = -Jcam * d_expy_d_y(y);
            //     _jacobianOplus[2] = Jcam * T_ca.rotation().toRotationMatrix() * d_expy_d_y(x_a);
            // }
            
            // void linearizeOplus()
            // {
            //     VertexSBAPointXYZ *vpoint = static_cast<VertexSBAPointXYZ *>(_vertices[0]);
            //     Vector3 psi_a = vpoint->estimate();
            //     VertexSE3Expmap *vpose = static_cast<VertexSE3Expmap *>(_vertices[1]);
            //     SE3Quat T_cw = vpose->estimate();
            //     VertexSE3Expmap *vanchor = static_cast<VertexSE3Expmap *>(_vertices[2]);
            //     const camParams *cam = static_cast<const camParams *>(parameter(0));

            //     SE3Quat A_aw = vanchor->estimate();
            //     SE3Quat T_ca = T_cw * A_aw.inverse();
            //     Vector3 x_a = invert_depth(psi_a);
            //     Vector3 y = T_ca * x_a;
            //     Matrix<number_t, 2, 3, Eigen::ColMajor> Jcam = d_proj_d_y(cam->focal_length, y);
            //     _jacobianOplus[0] = -Jcam * d_Tinvpsi_d_psi(T_ca, psi_a);
            //     _jacobianOplus[1] = -Jcam * d_expy_d_y(y);
            //     _jacobianOplus[2] = Jcam * T_ca.rotation().toRotationMatrix() * d_expy_d_y(x_a);
            // }
            camParams *_cam;
        };

    } // namespace OptimizationStructs

    HSLAM::Sim3 g2oSim3_to_sophusSim3(HSLAM::Sim3Vertex &g2o_sim3);
    g2o::Sim3 sophusSim3_to_g2oSim3(HSLAM::Sim3 sophus_sim3);

    bool PoseOptimization(std::shared_ptr<Frame> pFrame, CalibHessian *calib, bool updatePose = true);
    int checkOutliers(std::shared_ptr<Frame> pFrame, CalibHessian* calib);
    
    int OptimizeSim3(std::shared_ptr<Frame> pKF1, std::shared_ptr<Frame> pKF2, std::vector<std::shared_ptr<MapPoint>> &vpMatches1, Sim3 &g2oS12, const float th2, const bool bFixScale);
    // void OptimizeEssentialGraph(std::shared_ptr<Map> pMap, FullSystem* _fs, std::shared_ptr<Frame> pLoopKF, std::shared_ptr<Frame> pCurKF, const KeyFrameAndPose &NonCorrectedSim3, const KeyFrameAndPose &CorrectedSim3,
    //                             const std::map<std::shared_ptr<Frame>, std::set<std::shared_ptr<Frame>, std::owner_less<std::shared_ptr<Frame>>>, std::owner_less<std::shared_ptr<Frame>>> &LoopConnections, const bool &bFixScale);


    void OptimizeEssentialGraph(std::vector<FrameShell*> & vpKFs, std::vector<std::shared_ptr<MapPoint>> &allMapPoints,  std::set<std::shared_ptr<Frame>> &TempFixed, 
                                std::shared_ptr<Frame> pLoopKF, std::shared_ptr<Frame> pCurKF,
                                const KeyFrameAndPose &NonCorrectedSim3, const KeyFrameAndPose &CorrectedSim3,
                                const std::map<std::shared_ptr<Frame>, std::set<std::shared_ptr<Frame>, std::owner_less<std::shared_ptr<Frame>>>, std::owner_less<std::shared_ptr<Frame>>> &LoopConnections,
                                const std::map<uint64_t, Eigen::Vector2i, std::less<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>> &connectivity,
                                const size_t maxKfIdatCand, const size_t minActkfid, const size_t maxMPIdatCand, const bool &bFixScale);



    void GlobalBundleAdjustemnt(std::shared_ptr<Map> pMap, int nIterations, bool *pbStopFlag, const unsigned long nLoopKF, const bool bRobust, const bool useSchurTrick);
    // void BundleAdjustment(const std::vector<std::shared_ptr<Frame>> &vpKFs, const std::vector<std::shared_ptr<MapPoint>> &vpMP,
    //                       std::vector<std::shared_ptr<Frame>> &activeKfs, std::vector<std::shared_ptr<MapPoint>> &activeMps,
    //                       int nIterations, bool *pbStopFlag, const bool bRobust, const bool useSchurTrick,
    //                       int totalKfId, int currMaxKF, int currMaxMp);
    void BundleAdjustment(const std::vector<std::shared_ptr<Frame>> &vpKFs, const std::vector<std::shared_ptr<MapPoint>> &vpMP,
                          int nIterations, bool *pbStopFlag, const bool bRobust, const bool useSchurTrick,
                          const size_t maxKfIdatCand, const size_t minActkfid, const size_t maxMPIdatCand);
} // namespace HSLAM

#endif