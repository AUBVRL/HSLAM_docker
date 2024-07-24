#pragma once
#ifndef __Se3_H__
#define __Se3_H__

#include "util/NumType.h"
#include <g2o/core/base_vertex.h>
#include "g2o/core/base_binary_edge.h"
#include "g2o/core/parameter.h"
#include "util/globalFuncs.h"


namespace HSLAM
{
    

    class camParams : public g2o::Parameter
    { //for some reason the default cameraParameters use a single focal length fx! rewrite necessary classes for fx fy
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        camParams(double fx, double fy, double cx, double cy) //: focal_length(focal_length), principle_point(principle_point)
        {
            focal_length[0] = fx;
            focal_length[1] = fy;
            principle_point[0] = cx;
            principle_point[1] = cy;
        }
        Vec2 cam_map(const Vec3 &trans_xyz) const
        {
            Vec2 proj;
            proj[0] = trans_xyz[0] / trans_xyz[2];
            proj[1] = trans_xyz[1] / trans_xyz[2];
            Vec2 res;
            res[0] = proj[0] * focal_length[0] + principle_point[0];
            res[1] = proj[1] * focal_length[1] + principle_point[1];
            return res;
        }

        Vec3 camInvMap(const Vec2 &coords) const
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

    class SE3Vertex : public g2o::BaseVertex<6, SE3>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        SE3Vertex() : g2o::BaseVertex<6, SE3>() {}

        virtual bool read(std::istream &is) override
        {
            Vec6 cam2world;
            for (int i = 0; i < 6; i++)
                is >> cam2world[i];            
            setEstimate(SE3::exp(cam2world).inverse());
            return true;
        }

        virtual bool write(std::ostream &os) const
        {
            SE3 cam2world(estimate().inverse());
            Vec6 lv = cam2world.log();
            for (int i = 0; i < 6; i++)
                os << lv[i] << " ";
            return os.good();
        }

        virtual void setToOriginImpl() override
        {
            _estimate = SE3();
        }

        virtual void oplusImpl(const double *update_) override
        {
            Eigen::Map<Vec6> update(const_cast<double*>(update_));
            _estimate = SE3::exp(update) * _estimate;
        }
    };

    class VertexPointIDepth : public g2o::BaseVertex<1, double>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        VertexPointIDepth() {}
        bool read(std::istream &is)
        {
            is >> UV[0];
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

    class VertexSBAPointXYZ : public g2o::BaseVertex<3, Vec3>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        VertexSBAPointXYZ(){};
        virtual bool read(std::istream &is) override
        {
           return readVector(is, _estimate);
        }

        bool write(std::ostream &os) const
        {
            return writeVector(os, estimate());
        }

        virtual void setToOriginImpl()
        {
            _estimate.fill(0);
        }

        virtual void oplusImpl(const number_t *update)
        {
            Eigen::Map<const Vec3> v(update);
            _estimate += v;
        }
    };

    class EdgeProjectinvDepth : public g2o::BaseMultiEdge<2, Vec2>
    {
    public:
        //the following parameterization stores the map points in VertexSBAPointXYZ struct
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeProjectinvDepth()
        {
            resize(3);                    //set the number of vertices
            resizeParameters(1);          //set the number of parameters to a class (camParams) that contains all params
            installParameter(_cam, 0, 0); //register the camparams class must be defined in the function creating the edges
        }

        bool read(std::istream &is)
        {
            readParamIds(is);
            readVector(is, _measurement);
            return readInformationMatrix(is);
        }
        bool write(std::ostream &os) const
        {
            writeParamIds(os);
            writeVector(os, measurement());
            return writeInformationMatrix(os);
        }

        Vec3 invert_depth(const Vec3 &x)
        {
            Vec3 res = x;
            return res / x(2);
        }

        void computeError()
        {
            const VertexPointIDepth *psi = static_cast<const VertexPointIDepth *>(_vertices[0]);
            const SE3Vertex *T_p_from_world = static_cast<const SE3Vertex *>(_vertices[1]);
            const SE3Vertex *T_anchor_from_world = static_cast<const SE3Vertex *>(_vertices[2]);
            const HSLAM::camParams *cam = static_cast<const HSLAM::camParams *>(parameter(0));

            Vec2 obs(_measurement);
            // _error = obs - cam->cam_map(T_p_from_world->estimate() * T_anchor_from_world->estimate().inverse() * invert_depth(psi->estimate()));
            _error = obs - cam->cam_map( (T_p_from_world->estimate() * T_anchor_from_world->estimate().inverse()) * (cam->camInvMap(psi->UV) * (1.0/psi->estimate()))); //invert_depth
        }

        // void linearizeOplus()
        // {
        //     VertexPointIDepth *vpoint = static_cast<VertexPointIDepth *>(_vertices[0]);
        //     double idepth = vpoint->estimate();

        //     SE3Vertex *vpose = static_cast<SE3Vertex *>(_vertices[1]);
        //     SE3 T_cw = vpose->estimate();
        //     SE3Vertex *vanchor = static_cast<SE3Vertex *>(_vertices[2]);
        //     SE3 A_aw = vanchor->estimate();

        //     const HSLAM::camParams *cam = static_cast<const HSLAM::camParams *>(parameter(0));
        //     SE3 T_ca = T_cw * A_aw.inverse();
        //     Vec3 x_a = cam->camInvMap(vpoint->UV) * (1.0/idepth);

        //     Vec3 y = (T_ca * cam->camInvMap(vpoint->UV)) * (1.0 / idepth);
        //     Eigen::Matrix<number_t, 2, 3, Eigen::ColMajor> Jcam = d_proj_d_y(cam->focal_length, y);
        //     _jacobianOplus[0] = -Jcam * d_Tinvpsi_d_psi(T_ca, psi_a);

        //     _jacobianOplus[1] = -Jcam * d_expy_d_y(y);
        //     _jacobianOplus[2] = Jcam * T_ca.rotationMatrix() * d_expy_d_y(x_a);
        // }

        Eigen::Matrix<double, 2, 3, Eigen::ColMajor> d_proj_d_y(const Vec2 &f, const Vec3 &xyz)
        {
            double z_sq = xyz[2] * xyz[2];
            Eigen::Matrix<double, 2, 3, Eigen::ColMajor> J;
            J << f[0] / xyz[2], 0, -(f[0] * xyz[0]) / z_sq,
                0, f[1] / xyz[2], -(f[1] * xyz[1]) / z_sq;
            return J;
        }

        Eigen::Matrix<double, 3, 6, Eigen::ColMajor> d_expy_d_y(const Vec3 &y)
        {
            Eigen::Matrix<double, 3, 6, Eigen::ColMajor> J;
            J.topLeftCorner<3, 3>() = -g2o::skew(y);
            J.bottomRightCorner<3, 3>().setIdentity();
            return J;
        }

        // inline Mat33 d_Tinvpsi_d_psi(const SE3 &T, const Vec3 &psi)
        // {
        //     Mat33 R = T.rotationMatrix();
        //     Vec3 x = invert_depth(psi);
        //     Vec3 r1 = R.col(0);
        //     Vec3 r2 = R.col(1);
        //     Mat33 J;
        //     J.col(0) = r1;
        //     J.col(1) = r2;
        //     J.col(2) = -R * x;
        //     J *= 1. / psi.z();
        //     return J;
        // }

        HSLAM::camParams *_cam;
    };

    // inline Vec3 invert_depth(const Vec3 &x)
    // {
    //     Vec3 res = x;
    //     return res / x(2);
    // }

    Vec3 unproject2d(const Vec2 &v)
    {
        Vec3 res;
        res(0) = v(0);
        res(1) = v(1);
        res(2) = 1;
        return res;
    }

    inline Vec3 invert_depth(const Vec3 &x)
    {
        return unproject2d(x.head<2>()) / x[2];
    }

    class EdgeProjectPSI2UV : public g2o::BaseMultiEdge<2, Vec2>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeProjectPSI2UV()
        {
            resize(3);
            resizeParameters(1);
            installParameter(_cam, 0, 0);
        }

        virtual bool read(std::istream &is) override
        {
            readParamIds(is);
            readVector(is, _measurement);
            return readInformationMatrix(is);
        }
        virtual bool write(std::ostream &os) const
        {
            writeParamIds(os);
            writeVector(os, measurement());
            return writeInformationMatrix(os);
        }
        virtual void computeError() override
        {
            const VertexSBAPointXYZ *psi = static_cast<const VertexSBAPointXYZ *>(_vertices[0]);
            const g2o::VertexSE3Expmap *T_p_from_world = static_cast<const g2o::VertexSE3Expmap *>(_vertices[1]);
            const g2o::VertexSE3Expmap *T_anchor_from_world = static_cast<const g2o::VertexSE3Expmap *>(_vertices[2]);
            const HSLAM::camParams *cam = static_cast<const HSLAM::camParams *>(parameter(0));

            Vec2 obs(_measurement);
            _error = obs - cam->cam_map(T_p_from_world->estimate() * T_anchor_from_world->estimate().inverse() * HSLAM::invert_depth(psi->estimate()));
        }

        inline Eigen::Matrix<double, 2, 3, Eigen::ColMajor> d_proj_d_y(const Vec2 &f, const Vec3 &xyz)
            {
                double z_sq = xyz[2] * xyz[2];
                Eigen::Matrix<double, 2, 3, Eigen::ColMajor> J;
                J << f[0] / xyz[2], 0, -(f[0] * xyz[0]) / z_sq,
                    0, f[1] / xyz[2], -(f[1] * xyz[1]) / z_sq;
                return J;
            }

        inline Eigen::Matrix<number_t, 3, 6, Eigen::ColMajor> d_expy_d_y(const Vec3 &y)
        {
            Eigen::Matrix<number_t, 3, 6, Eigen::ColMajor> J;
            J.topLeftCorner<3, 3>() = -g2o::skew(y);
            J.bottomRightCorner<3, 3>().setIdentity();

            return J;
        }

        inline Eigen::Matrix<number_t, 3, 3, Eigen::ColMajor> d_Tinvpsi_d_psi(const g2o::SE3Quat &T, const Vec3 &psi)
        {
            Mat33 R = T.rotation().toRotationMatrix();
            Vec3 x = HSLAM::invert_depth(psi);
            Vec3 r1 = R.col(0);
            Vec3 r2 = R.col(1);
            Mat33 J;
            J.col(0) = r1;
            J.col(1) = r2;
            J.col(2) = -R * x;
            J *= 1. / psi.z();
            return J;
        }

        

        virtual void linearizeOplus() override
        {
            VertexSBAPointXYZ *vpoint = static_cast<VertexSBAPointXYZ *>(_vertices[0]);
            Vec3 psi_a = vpoint->estimate();
            g2o::VertexSE3Expmap *vpose = static_cast<g2o::VertexSE3Expmap *>(_vertices[1]);
            g2o::SE3Quat T_cw = vpose->estimate();
            g2o::VertexSE3Expmap *vanchor = static_cast<g2o::VertexSE3Expmap *>(_vertices[2]);
            const HSLAM::camParams *cam = static_cast<const HSLAM::camParams *>(parameter(0));

            g2o::SE3Quat A_aw = vanchor->estimate();
            g2o::SE3Quat T_ca = T_cw * A_aw.inverse();
            Vec3 x_a = HSLAM::invert_depth(psi_a);
            Vec3 y = T_ca * x_a;
            Eigen::Matrix<number_t, 2, 3, Eigen::ColMajor> Jcam = d_proj_d_y(cam->focal_length, y);
            _jacobianOplus[0] = -Jcam * d_Tinvpsi_d_psi(T_ca, psi_a);
            _jacobianOplus[1] = -Jcam * d_expy_d_y(y);
            _jacobianOplus[2] = Jcam * T_ca.rotation().toRotationMatrix() * d_expy_d_y(x_a);
        }
        HSLAM::camParams *_cam;
    };

} // namespace HSLAM

#endif
