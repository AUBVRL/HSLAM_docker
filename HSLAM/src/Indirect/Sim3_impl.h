#pragma once
#ifndef __Sim3_H__
#define __Sim3_H__

#include "util/NumType.h"
#include <g2o/core/base_vertex.h>
#include "g2o/core/base_binary_edge.h"
#include "g2o/types/sim3/sim3.h"
#include "util/globalFuncs.h"

namespace HSLAM
{
    class VertexXYZPt : public g2o::BaseVertex<3, Vec3>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        VertexXYZPt(){}
        virtual bool read(std::istream &is) override
        {
            return readVector(is, _estimate);
        }
        virtual bool write(std::ostream &os) const
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

    class Sim3Vertex : public g2o::BaseVertex<7, g2o::Sim3>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        Sim3Vertex() : g2o::BaseVertex<7, g2o::Sim3>() {}

        void setData(double _fx, double _fy, double _cx, double _cy, bool _fixScale = false)
        {
            fx = _fx;
            fy = _fy;
            cx = _cx;
            cy = _cy;
            fixScale = _fixScale;
            _marginalized = false;
        }

        void setData2(double _fx, double _fy, double _cx , double _cy)
        {
            fx2 = _fx;
            fy2 = _fy;
            cx2 = _cx;
            cy2 = _cy;
        }

        virtual bool read(std::istream &is) override
        {
            Vec7 cam2world;
            for (int i = 0; i < 7; i++)
                is >> cam2world[i];
            is >> fx; is >> fy; is >> cx; is >> cy; is >> fixScale;
            is >> fx2; is >> fy2; is >> cx2; is >> cy2;
            setEstimate(g2o::Sim3(cam2world).inverse());
            return true;
        }

        virtual bool write(std::ostream &os) const
        {
            g2o::Sim3 cam2world(estimate().inverse()); //    estimate().inverse());
            Vec7 lv = cam2world.log();
            for (int i = 0; i < 7; i++)
                os << lv[i] << " ";
            os << fx << " " << fy << " " << cx << " " << cy << " " << fixScale << " ";
            os << fx2 << " " << fy2 << " " << cx2 << " " << cy2 << " ";
            return os.good();
        }

        virtual void setToOriginImpl() override
        {
            _estimate = g2o::Sim3();
        }

        virtual void oplusImpl(const double *update_) override
        {
            Eigen::Map<Vec7> update(const_cast<double*>(update_));
            if (fixScale)
                update[6] = 0;
            // // std::cout<<update[6]<<std::endl;
            // if(update[6] < -1e-3){
            //     _is_invalid = true;
            //     update[6] = 0.;
            // }
            _estimate = g2o::Sim3(update) * _estimate;
        }

        Vec2 cam_map(const Vec2 &v) const
        {
            return Vec2(v[0] * fx + cx, v[1] * fy + cy);
        }

        Vec2 cam_map2(const Vec2 &v) const
        {
            return Vec2(v[0] * fx2 + cx2, v[1] * fy2 + cy2);
        }

        bool fixScale = false;
        bool _is_invalid = false;
        double cx, cy, fx, fy;
        double cx2, cy2, fx2, fy2;

    };

    class EdgeSim3ProjectXYZ : public g2o::BaseBinaryEdge<2, Vec2, VertexXYZPt, Sim3Vertex>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        EdgeSim3ProjectXYZ(){};
        virtual bool read(std::istream &is);
        virtual bool write(std::ostream &os) const;

        void computeError()
        {
            const Sim3Vertex *v1 = static_cast<const Sim3Vertex *>(_vertices[1]);
            const VertexXYZPt *v2 = static_cast<const VertexXYZPt *>(_vertices[0]);

            Vec2 obs(_measurement);
            _error = obs - v1->cam_map( project( v1->estimate().map(v2->estimate())));
        }
        // virtual void linearizeOplus();
    };


    bool EdgeSim3ProjectXYZ::read(std::istream &is)
    {
        for (int i = 0; i < 2; i++)
            is >> _measurement[i];
        is >> information()(0, 0);
        is >> information()(0, 1);
        is >> information()(1, 1);
        information()(1, 0) = information()(0, 1);
        return true;
    }

    bool EdgeSim3ProjectXYZ::write(std::ostream &os) const
    {
        for (int i = 0; i < 2; i++)
        {
            os << _measurement[i] << " ";
        }

        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
            {
                os << " " << information()(i, j);
            }
        return os.good();
    }

    class  EdgeInverseSim3ProjectXYZ : public g2o::BaseBinaryEdge<2, Vec2,  VertexXYZPt, Sim3Vertex>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        EdgeInverseSim3ProjectXYZ(){};
        virtual bool read(std::istream &is) override
        {
            for (int i = 0; i < 2; i++)
                is >> _measurement[i];
            is >> information()(0, 0);
            is >> information()(0, 1);
            is >> information()(1, 1);
            information()(1, 0) = information()(0, 1);
            return true;
        }
        virtual bool write(std::ostream &os) const
        {
            for (int i = 0; i < 2; i++)
            {
                os << _measurement[i] << " ";
            }

            for (int i = 0; i < 2; i++)
                for (int j = i; j < 2; j++)
                {
                    os << " " << information()(i, j);
                }
            return os.good();
        }

        void computeError()
        {
            const Sim3Vertex *v1 = static_cast<const Sim3Vertex *>(_vertices[1]);
            const VertexXYZPt *v2 = static_cast<const VertexXYZPt *>(_vertices[0]);

            Vec2 obs(_measurement);
            _error = obs - v1->cam_map2(project(v1->estimate().inverse().map(v2->estimate())));
        }
        // virtual void linearizeOplus();
    };


    class EdgeSim3 : public g2o::BaseBinaryEdge<7, g2o::Sim3, Sim3Vertex, Sim3Vertex>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        EdgeSim3(){}
        virtual bool read(std::istream &is) override
        {
            Vec7 v7;
            for (int i = 0; i < 7; i++)
                is >> v7[i];
            setMeasurement(g2o::Sim3(v7).inverse());

            for (int i = 0; i < 7; i++)
                for (int j = i; j < 7; j++)
                {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            return true;
        }
        virtual bool write(std::ostream &os) const
        {
            g2o::Sim3 cam2world(measurement().inverse());
            Vec7 v7 = cam2world.log();
            for (int i = 0; i < 7; i++)
            {
                os << v7[i] << " ";
            }
            for (int i = 0; i < 7; i++)
                for (int j = i; j < 7; j++)
                {
                    os << " " << information()(i, j);
                }
            return os.good();
        }

        void computeError()
        {
            const Sim3Vertex *v1 = static_cast<const Sim3Vertex *>(_vertices[0]);
            const Sim3Vertex *v2 = static_cast<const Sim3Vertex *>(_vertices[1]);

            g2o::Sim3 C(_measurement);
            g2o::Sim3 error_ = C * v1->estimate() * v2->estimate().inverse();
            _error = error_.log();
        }

        virtual number_t initialEstimatePossible(const g2o::OptimizableGraph::VertexSet &, g2o::OptimizableGraph::Vertex *) { return 1.; }
        virtual void initialEstimate(const g2o::OptimizableGraph::VertexSet &from, g2o::OptimizableGraph::Vertex * /*to*/)
        {
            Sim3Vertex *v1 = static_cast<Sim3Vertex *>(_vertices[0]);
            Sim3Vertex *v2 = static_cast<Sim3Vertex *>(_vertices[1]);
            if (from.count(v1) > 0)
                v2->setEstimate(measurement() * v1->estimate());
            else
                v1->setEstimate(measurement().inverse() * v2->estimate());
        }
        // virtual void linearizeOplus();
    };

} // namespace HSLAM
#endif