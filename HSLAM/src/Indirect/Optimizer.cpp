#include "Indirect/Optimizer.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>

#include <g2o/solvers/dense/linear_solver_dense.h>
// #include "g2o/solvers/linear_solver_dense.h"

#include <g2o/solvers/eigen/linear_solver_eigen.h>
// #include <g2o/solvers/linear_solver_eigen.h>

#include <boost/thread.hpp>
#include "Indirect/Frame.h"
#include "util/FrameShell.h"
#include "Indirect/MapPoint.h"
#include "Indirect/Map.h"
#include "FullSystem/FullSystem.h"


#include "Indirect/Sim3_impl.h"
#include "Indirect/Se3.h"
// #include <g2o/types/sim3/types_seven_dof_expmap.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include "g2o/solvers/cholmod/linear_solver_cholmod.h"

// #include "g2o/types/types_seven_dof_expmap.h"


namespace HSLAM
{
    using namespace std;
    using namespace OptimizationStructs;

    HSLAM::Sim3 g2oSim3_to_sophusSim3(Sim3Vertex &g2o_sim3)
    {
        Mat44 sim_transform;
        sim_transform.topLeftCorner(3, 3) = g2o_sim3.estimate().rotation().toRotationMatrix();
        sim_transform.topRightCorner(3, 1) = g2o_sim3.estimate().translation();
        sim_transform.topLeftCorner(3, 3) *= g2o_sim3.estimate().scale();
        return Sim3(sim_transform);
    }

    g2o::Sim3 sophusSim3_to_g2oSim3(HSLAM::Sim3 sophus_sim3)
    {
        return g2o::Sim3(sophus_sim3.rotationMatrix(), sophus_sim3.translation(), sophus_sim3.scale());
    }

    bool PoseOptimization(std::shared_ptr<Frame> pFrame, CalibHessian *calib, bool updatePose)
    {
        g2o::SparseOptimizer optimizer;
        auto linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
        optimizer.setAlgorithm(solver);

        int nInitialCorrespondences = 0;

        // Set Frame vertex
        vertexSE3 *vSE3 = new vertexSE3(); //VertexSE3Expmap Converter::toSE3Quat(pFrame->mTcw));
        vSE3->setEstimate(pFrame->fs->getPoseInverse());
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);
        // Set MapPoint vertices
        const int N = pFrame->nFeatures;

        vector<edgeSE3XYZPoseOnly *> vpEdgesMono;
        vector<size_t> vnIndexEdgeMono;
        vector<double> initErr;
        double initScale = 1.0;
        initErr.reserve(2 * N);
        vpEdgesMono.reserve(N);
        vnIndexEdgeMono.reserve(N);
        vector<double> vInformation;
        vInformation.reserve(N);
        double maxInfo = 1e7;
        double stdDev = 1e7;

        const float deltaMono = sqrt(1.345); //sqrt(1.345); // sqrt(5.991);
        {
            // boost::unique_lock<boost::mutex> lock(MapPoint::mGlobalMutex); //this would lock ALL map points poses from changing!

            for (int i = 0; i < N; i++)
            {
                std::shared_ptr<MapPoint> pMP = pFrame->tMapPoints[i];
                if (pMP)
                {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    edgeSE3XYZPoseOnly *e = new edgeSE3XYZPoseOnly();
                    e->setCamera(calib->fxl(), calib->fyl(), calib->cxl(), calib->cyl());
                    e->setXYZ(pMP->getWorldPose().cast<double>());
                    e->setInformation(Eigen::Matrix2d::Identity());
                    e->setMeasurement(Vec2((double)pFrame->mvKeys[i].pt.x, (double)pFrame->mvKeys[i].pt.y));
                    e->setId(i);

                    e->setVertex(0, vSE3);

                    e->computeError();
                    initErr.push_back(e->error()[0]);
                    initErr.push_back(e->error()[1]);

                    double Info = sqrtf(pMP->getidepthHessian());
                    vInformation.push_back(Info);
                    if (Info > maxInfo)
                        maxInfo = Info;

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);

                    rk->setDelta(deltaMono);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
            }

            //compute information vector distribution:
            if (normalizeInfoWithVariance)
                stdDev = getStdDev(vInformation);

            initScale = getStdDev(initErr); //computeScale(initErr);

            for (int i = 0, iend = vpEdgesMono.size(); i < iend; ++i)
            {
                vpEdgesMono[i]->setScale(initScale);

                if (normalizeInfoWithVariance)
                    vpEdgesMono[i]->setInformation(Eigen::Matrix2d::Identity() * (vInformation[i] / (stdDev + 0.00001)));  //* invSigma2); //set this to take into account depth variance!
                else                                                                                                       //normalizing by the maximum
                    vpEdgesMono[i]->setInformation(Eigen::Matrix2d::Identity() * (vInformation[i] / (maxInfo + 0.00001))); //* invSigma2); //set this to take into account depth variance!
            }
        }
        if (nInitialCorrespondences < 10 || optimizer.edges().size() < 10)
            return false;

        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        // const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991}; //1.345
        const float chi2Mono[4] = {1.345, 1.345, 1.345, 1.345}; //5.991
        const int its[4] = {10, 10, 10, 10};

        int nBad = 0;
        for (size_t it = 0; it < 4; it++)
        {
            vSE3->setEstimate(pFrame->fs->getPoseInverse());
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);
            nBad = 0;
            for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
            {
                edgeSE3XYZPoseOnly *e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];

                if (pFrame->mvbOutlier[idx])
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if (chi2 > chi2Mono[it])
                {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                }
                else
                {
                    pFrame->mvbOutlier[idx] = false;
                    e->setLevel(0);
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

            if (optimizer.edges().size() < 10)
                break;

            if (!updatePose)
                break;
        }

        number_t *hessianData = vSE3->hessianData();
        Vec6 vHessian;
        vHessian<< hessianData[0], hessianData[7], hessianData[14], hessianData[21], hessianData[28], hessianData[35];

        bool isUsable = vHessian.norm() > 1e6;
        // // Recover optimized pose
        if(isUsable && updatePose)
            pFrame->fs->setPose(vSE3->estimate().inverse());
        // bool isUsable = false;
        return isUsable;
    }

    int checkOutliers(std::shared_ptr<Frame> pFrame, CalibHessian* calib)
    {
        g2o::SparseOptimizer optimizer;
        auto linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
        optimizer.setAlgorithm(solver);

        vertexSE3 *vSE3 = new vertexSE3(); //VertexSE3Expmap Converter::toSE3Quat(pFrame->mTcw));
        vSE3->setEstimate(pFrame->fs->getPoseInverse());

        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);

        const int N = pFrame->nFeatures;
        vector<edgeSE3XYZPoseOnly *> vpEdgesMono;
        vector<size_t> vnIndexEdgeMono;
        vector<double> initErr;
        double initScale = 1.0;
        initErr.reserve(2 * N);
        vpEdgesMono.reserve(N);
        vnIndexEdgeMono.reserve(N);
        vector<double> vInformation;
        vInformation.reserve(N);
        double stdDev = 1e7;

        for (int i = 0; i < N; i++)
        {
            std::shared_ptr<MapPoint> pMP = pFrame->tMapPoints[i];
            if (pMP)
            {
                pFrame->mvbOutlier[i] = false;

                edgeSE3XYZPoseOnly *e = new edgeSE3XYZPoseOnly();
                e->setCamera(calib->fxl(), calib->fyl(), calib->cxl(), calib->cyl());
                e->setXYZ(pMP->getWorldPose().cast<double>());
                e->setInformation(Eigen::Matrix2d::Identity());
                e->setMeasurement(Vec2((double)pFrame->mvKeys[i].pt.x, (double)pFrame->mvKeys[i].pt.y));
                e->setId(i);

                e->setVertex(0, vSE3);

                e->computeError();
                initErr.push_back(e->error()[0]);
                initErr.push_back(e->error()[1]);

                double Info = pMP->getidepthHessian();
                vInformation.push_back(Info);
                optimizer.addEdge(e);
                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);
            }
        }

        stdDev = getStdDev(vInformation);
        initScale = getStdDev(initErr);
        for (int i = 0, iend = vpEdgesMono.size(); i < iend; ++i)
        {
            // vpEdgesMono[i]->setScale(initScale);
            vpEdgesMono[i]->setInformation(Eigen::Matrix2d::Identity() * (vInformation[i] / (stdDev + 0.00001))); //* invSigma2); //set this to take into account depth variance!
        }

        optimizer.initializeOptimization(0);
        int nBad = 0;
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            edgeSE3XYZPoseOnly *e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if (pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if (chi2 > 5.991)
            {
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx] = false;
                e->setLevel(0);
            }
        }

        return nBad;
    }

int OptimizeSim3(std::shared_ptr<Frame> pKF1, std::shared_ptr<Frame> pKF2, std::vector<std::shared_ptr<MapPoint>> &vpMatches1, Sim3 &g2oS12, const float th2, const bool bFixScale)
{

    g2o::SparseOptimizer optimizer;
    auto linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver)));
    optimizer.setAlgorithm(solver);
    
    auto PKF1Pose = pKF1->fs->getPoseOpti();
    auto R1w = PKF1Pose.rotationMatrix();
    auto t1w = PKF1Pose.translation();

    auto PKF2Pose = pKF2->fs->getPoseOpti();  
    auto R2w = PKF2Pose.rotationMatrix();
    auto t2w = PKF2Pose.translation();

    size_t id = 0;
    // Set Sim3 vertex
    Sim3Vertex* vSim3 = new Sim3Vertex();
    vSim3->setData(pKF1->HCalib->fxl(), pKF1->HCalib->fyl(), pKF1->HCalib->cxl(), pKF1->HCalib->cyl(), bFixScale);
    vSim3->setData2(pKF2->HCalib->fxl(), pKF2->HCalib->fyl(), pKF2->HCalib->cxl(), pKF2->HCalib->cyl());
    vSim3->setId(id);
    vSim3->setFixed(false);
    vSim3->setEstimate(g2o::Sim3(g2oS12.rotationMatrix(), g2oS12.translation(), g2oS12.scale()));
    optimizer.addVertex(vSim3);
    id++;
    // Set MapPoint vertices
    const int N = vpMatches1.size();
    const vector<std::shared_ptr<MapPoint>> vpMapPoints1 = pKF1->getMapPointsV();
    vector<EdgeSim3ProjectXYZ*> vpEdges12;
    vector<EdgeInverseSim3ProjectXYZ*> vpEdges21;
    vector<size_t> vnIndexEdge;

    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;

    for(int i=0; i<N; i++)
    {
        if(!vpMatches1[i])
            continue;

        std::shared_ptr<MapPoint> pMP1 = vpMapPoints1[i];
        std::shared_ptr<MapPoint> pMP2 = vpMatches1[i];

        size_t id1 = id;
        size_t id2 = id + 1;
        id = id + 2;

        const int i2 = pMP2->getIndexInKF(pKF2);

        if(pMP1 && pMP2)
        {
            if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
            {
                VertexXYZPt* vPoint1 = new VertexXYZPt();
                Vec3 P3D1w = pMP1->getWorldPose().cast<double>();
                Vec3 P3D1c = R1w * P3D1w + t1w;
              
                vPoint1->setEstimate(P3D1c);
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                VertexXYZPt* vPoint2 = new VertexXYZPt();
                Vec3 P3D2w = pMP2->getWorldPose().cast<double>();
                Vec3 P3D2c = R2w * P3D2w + t2w;
                vPoint2->setEstimate(P3D2c);
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
                continue;
        }
        else
            continue;

        nCorrespondences++;

        // Set edge x1 = S12*X2
        Vec2 obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeys[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        EdgeSim3ProjectXYZ* e12 = new EdgeSim3ProjectXYZ();
        e12->setVertex(0,optimizer.vertex(id2)); // dynamic_cast<g2o::OptimizableGraph::Vertex*>()
        e12->setVertex(1, optimizer.vertex(0)); //dynamic_cast<g2o::OptimizableGraph::Vertex*>()
        e12->setMeasurement(obs1);

        //const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity()); //*invSigmaSquare1

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        Vec2 obs2;
        const cv::KeyPoint &kpUn2 = pKF2->mvKeys[i2];
        obs2 << kpUn2.pt.x, kpUn2.pt.y;

        EdgeInverseSim3ProjectXYZ* e21 = new EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, optimizer.vertex(id1)); //dynamic_cast<g2o::OptimizableGraph::Vertex*>()
        e21->setVertex(1, optimizer.vertex(0)); //dynamic_cast<g2o::OptimizableGraph::Vertex*>()
        e21->setMeasurement(obs2);
        // float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity()); //*invSigmaSquare2

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);
    }

    // Optimize!
    optimizer.initializeOptimization(0);
    optimizer.optimize(10);
    if(static_cast<Sim3Vertex*>(optimizer.vertex(0))->_is_invalid)
        return 0;

    // Check inliers
    int nBad=0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx] = nullptr;
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i]=static_cast<EdgeSim3ProjectXYZ*>(NULL);
            vpEdges21[i]=static_cast<EdgeInverseSim3ProjectXYZ*>(NULL);
            nBad++;
        }
    }

    int nMoreIterations;
    if(nBad>0)
        nMoreIterations=10;
    else
        nMoreIterations=5;

    if(nCorrespondences-nBad<10)
        return 0;

    // Optimize again only with inliers

    optimizer.initializeOptimization(0);
    optimizer.optimize(nMoreIterations);
    if(static_cast<Sim3Vertex*>(optimizer.vertex(0))->_is_invalid)
        return 0;

    int nIn = 0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx] = nullptr;
        }
        else
            nIn++;
    }

    // Recover optimized Sim3
    Sim3Vertex* vSim3_recov = static_cast<Sim3Vertex*>(optimizer.vertex(0));
    g2oS12 = g2oSim3_to_sophusSim3(*vSim3_recov);
    // g2oS12 = Sim3(vSim3_recov->estimate().rotation().toRotationMatrix().to ;

    return nIn;
}


void OptimizeEssentialGraph(std::vector<FrameShell*> & vpKFs, std::vector<std::shared_ptr<MapPoint>> &vpMPs, std::set<std::shared_ptr<Frame>> &TempFixed, 
                            std::shared_ptr<Frame> pLoopKF, std::shared_ptr<Frame> pCurKF,
                            const KeyFrameAndPose &NonCorrectedSim3, const KeyFrameAndPose &CorrectedSim3, 
                            const std::map<std::shared_ptr<Frame>, std::set<std::shared_ptr<Frame>, std::owner_less<std::shared_ptr<Frame>>>, std::owner_less<std::shared_ptr<Frame>>> &LoopConnections,
                            const std::map<uint64_t, Eigen::Vector2i, std::less<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>> &connectivity,
                            const size_t maxKfIdatCand, const size_t minActkfid,const size_t maxMPIdatCand, const bool &bFixScale)
        {

            g2o::SparseOptimizer optimizer;
            g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolver_7_3>(g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>>()));
            solver->setUserLambdaInit(1e-16);
            optimizer.setAlgorithm(solver);

            const size_t nMaxKFid = vpKFs.size();
            std::vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw(nMaxKFid + 1);
            std::map<int, std::shared_ptr<Frame>> keyframesByKFID;

            const int minFeat = 100;

            //grab any valid calibration 
            CalibHessian *hcalib = nullptr; 
            int attempts = 0;
            while (hcalib == nullptr )
            {
                if(attempts > 500) break;
                auto goodKF = vpKFs[attempts]->frame;
                if(!goodKF) { attempts++; continue;}
                hcalib = goodKF->HCalib;
            }

            // Set KeyFrame vertices
            for (size_t i = 0, iend = vpKFs.size(); i < iend; i++)
            {
                if(vpKFs[i]->KfId > maxKfIdatCand)
                    continue;

                const int nIDi = vpKFs[i]->KfId;

                std::shared_ptr<Frame> pKF = vpKFs[i]->frame;
                
                keyframesByKFID[nIDi] = pKF;

                Sim3Vertex *VSim3 = new Sim3Vertex();
                auto TempSiw = sophusSim3_to_g2oSim3(vpKFs[i]->getPoseOpti());
                vScw[nIDi] = TempSiw; //Siw
                VSim3->setEstimate(TempSiw);
                VSim3->setFixed(false);
                VSim3->setId(nIDi);
                VSim3->setData(hcalib->fxl(), hcalib->fyl(), hcalib->cxl(), hcalib->cyl());
                VSim3->setData2(hcalib->fxl(), hcalib->fyl(), hcalib->cxl(), hcalib->cyl());
                optimizer.addVertex(VSim3);

                if(!pKF)
                    continue;

                KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);
                
                // if (it != CorrectedSim3.end() && !TempFixed.count(pKF))
                //     VSim3->setFixed(true);

                if (pKF == pLoopKF )
                    VSim3->setFixed(true);

            

                if(pKF->fs->KfId >= minActkfid - 5)
                    VSim3->setFixed(true);

                if(TempFixed.count(pKF) >= 0 && pKF->fs->KfId > minActkfid)
                    VSim3->setFixed(false);
            
            }

            std::set<pair<long unsigned int, long unsigned int>> sInsertedEdges;

            const Mat77 matLambda = Mat77::Identity();
            int index = nMaxKFid + 1;
            // Set Loop edges
            for (auto mit = LoopConnections.begin(), mend = LoopConnections.end(); mit != mend; mit++)
            {
                std::shared_ptr<Frame> pKF = mit->first;

                const long unsigned int nIDi = pKF->fs->KfId;
                const std::set<std::shared_ptr<Frame>, std::owner_less<std::shared_ptr<Frame>>> &spConnections = mit->second;
                const g2o::Sim3 Siw = vScw[nIDi];
                const g2o::Sim3 Swi = Siw.inverse();

                for (std::set<std::shared_ptr<Frame>>::const_iterator sit = spConnections.begin(), send = spConnections.end(); sit != send; sit++)
                {
                    const long unsigned int nIDj = (*sit)->fs->KfId;
                    if ((nIDi != pCurKF->fs->KfId || nIDj != pLoopKF->fs->KfId) && pKF->GetWeight(*sit) < minFeat)
                        continue;

                    const g2o::Sim3 Sjw = vScw[nIDj];
                    const g2o::Sim3 Sji = Sjw * Swi;

                    EdgeSim3 *e = new EdgeSim3();
                    e->setId(index);
                    index++;
                    e->setVertex(1, optimizer.vertex(nIDj)); //dynamic_cast<g2o::OptimizableGraph::Vertex *>()
                    e->setVertex(0, optimizer.vertex(nIDi)); //dynamic_cast<g2o::OptimizableGraph::Vertex*>()
                    e->setMeasurement(Sji);
                    e->setInformation(matLambda);

                    optimizer.addEdge(e);

                    sInsertedEdges.insert(make_pair(min(nIDi, nIDj), max(nIDi, nIDj)));
                }
            }

            // Set normal edges
            for (size_t i = 0, iend = vpKFs.size(); i < iend; i++)
            {
                const int nIDi = vpKFs[i]->KfId;

                if(nIDi > maxKfIdatCand)
                    continue;
                
                std::shared_ptr<Frame> pKF = vpKFs[i]->frame;
                if(!pKF)
                    continue;

                g2o::Sim3 tSwi;

                KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

                if (iti != NonCorrectedSim3.end())
                    tSwi = sophusSim3_to_g2oSim3((iti->second).inverse());
                else
                    tSwi = vScw[nIDi].inverse();

                g2o::Sim3 Swi = tSwi;

                std::shared_ptr<Frame> pParentKF = pKF->GetParent();

                // Spanning tree edge
                if (pParentKF)
                {
                    int nIDj = pParentKF->fs->KfId;

                    g2o::Sim3 tSjw;

                    KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

                    if (itj != NonCorrectedSim3.end())
                        tSjw = sophusSim3_to_g2oSim3(itj->second);
                    else
                        tSjw = vScw[nIDj];

                    g2o::Sim3 Sjw = tSjw;

                    g2o::Sim3 Sji = Sjw * Swi;

                    EdgeSim3 *e = new EdgeSim3();
                    e->setId(index);
                    index++;
                    e->setVertex(1, optimizer.vertex(nIDj)); //dynamic_cast<g2o::OptimizableGraph::Vertex*>()
                    e->setVertex(0, optimizer.vertex(nIDi)); //dynamic_cast<g2o::OptimizableGraph::Vertex*>()
                    e->setMeasurement(Sji);

                    e->setInformation(matLambda);
                    optimizer.addEdge(e);
                    sInsertedEdges.insert(make_pair(min(nIDi, nIDj), max(nIDi, nIDj)));
                }

                // Loop edges
                const set<std::shared_ptr<Frame>> sLoopEdges = pKF->GetLoopEdges();
                for (std::set<std::shared_ptr<Frame>>::const_iterator sit = sLoopEdges.begin(), send = sLoopEdges.end(); sit != send; sit++)
                {
                    std::shared_ptr<Frame> pLKF = *sit;
                    if (pLKF->fs->KfId < pKF->fs->KfId)
                    {
                        g2o::Sim3 tSlw;

                        KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                        if (itl != NonCorrectedSim3.end())
                            tSlw = sophusSim3_to_g2oSim3(itl->second);
                        else
                            tSlw = vScw[pLKF->fs->KfId];

                       g2o::Sim3 Slw = tSlw;

                        g2o::Sim3 Sli = Slw * Swi;
                        EdgeSim3 *el = new EdgeSim3();
                        el->setId(index);
                        index++;
                        el->setVertex(1, optimizer.vertex(pLKF->fs->KfId)); //dynamic_cast<g2o::OptimizableGraph::Vertex*>()
                        el->setVertex(0, optimizer.vertex(nIDi));           //dynamic_cast<g2o::OptimizableGraph::Vertex*>()
                        el->setMeasurement(Sli);
                        el->setInformation(matLambda);

                        optimizer.addEdge(el);
                        sInsertedEdges.insert(make_pair(min(nIDi, (int)pLKF->fs->KfId), max(nIDi, (int)pLKF->fs->KfId)));
                    }
                }

                // Covisibility graph edges
                const vector<std::shared_ptr<Frame>> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
                for (vector<std::shared_ptr<Frame>>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end(); vit++)
                {
                    std::shared_ptr<Frame> pKFn = *vit;
                    if (pKFn && pKFn != pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
                    {
                        if (!pKFn->isBad() && pKFn->fs->KfId < pKF->fs->KfId)
                        {
                            if (sInsertedEdges.count(make_pair(std::min(pKF->fs->KfId, pKFn->fs->KfId), std::max(pKF->fs->KfId, pKFn->fs->KfId))))
                                continue;

                            g2o::Sim3 tSnw;

                            KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                            if (itn != NonCorrectedSim3.end())
                                tSnw = sophusSim3_to_g2oSim3(itn->second);
                            else
                                tSnw = vScw[pKFn->fs->KfId];

                            g2o::Sim3 Snw = tSnw;

                            g2o::Sim3 Sni = Snw * Swi;

                            EdgeSim3 *en = new EdgeSim3();
                            en->setId(index);
                            index++;
                            en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFn->fs->KfId)));
                            en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                            en->setMeasurement(Sni);
                            en->setInformation(matLambda);

                            optimizer.addEdge(en);
                            sInsertedEdges.insert(make_pair(min(nIDi, (int)pKFn->fs->KfId), max(nIDi, (int)pKFn->fs->KfId)));
                        }
                    }
                }
            }

            //setup Pose-Pose constraints
            for (std::pair<uint64_t, Eigen::Vector2i> p : connectivity)
            {
                int host = (int)(p.first >> 32);
                int target = (int)(p.first & (uint64_t)0xFFFFFFFF);
                assert(host >= 0 && target >= 0);

                if (host > maxKfIdatCand || target > maxKfIdatCand || host == target || host > target)
                    continue;
                if (sInsertedEdges.count(make_pair(std::min(host, target), std::max(host, target)))) //check that we did not add this edge yet.
                    continue;
                
                //set host
                g2o::Sim3 tSwi;
                KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(keyframesByKFID[host]);
                if (iti != NonCorrectedSim3.end())
                    tSwi = sophusSim3_to_g2oSim3((iti->second).inverse());
                else
                    tSwi = vScw[host].inverse();

                g2o::Sim3 Swi = tSwi;

                //set target pose
                g2o::Sim3 tSnw;
                KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(keyframesByKFID[target]);
                if (itn != NonCorrectedSim3.end())
                    tSnw = sophusSim3_to_g2oSim3(itn->second);
                else
                    tSnw = vScw[target];

                g2o::Sim3 Snw = tSnw;

                //set measurement
                g2o::Sim3 Sni = Snw * Swi;

                EdgeSim3 *en = new EdgeSim3();
                en->setId(index);
                index++;
                en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(target)));
                en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(host)));
                en->setMeasurement(Sni);
                en->setInformation(matLambda);
                optimizer.addEdge(en);
            }

            // Optimize!
            optimizer.initializeOptimization(0);
            optimizer.optimize(25);

            for (size_t i = 0; i < vpKFs.size(); i++)
            {
                const int nIDi = vpKFs[i]->KfId;
                if(nIDi > maxKfIdatCand)
                    continue;

                Sim3Vertex *VSim3 = static_cast<Sim3Vertex *>(optimizer.vertex(nIDi));
                vpKFs[i]->setPoseOpti(g2oSim3_to_sophusSim3(*VSim3));
                // vpKFs[i]->setPoseOpti(VSim3->estimate());
            }

            // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
            for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
            {
                std::shared_ptr<MapPoint> pMP = vpMPs[i];
                
                auto dirStat = pMP->getDirStatus();
                if (dirStat == MapPoint::active || dirStat == MapPoint::removed)
                    continue;

                if (pMP->isBad())
                    continue;

                pMP->updateGlobalPose();
                pMP->UpdateNormalAndDepth();
            }
}


// void GlobalBundleAdjustemnt(std::shared_ptr<Map> pMap, int nIterations, bool *pbStopFlag, const unsigned long nLoopKF, const bool bRobust, const bool useSchurTrick)
// {
//     std::vector<std::shared_ptr<Frame>> vpKFs;
//     pMap->GetAllKeyFrames(vpKFs);
//     std::vector<std::shared_ptr<MapPoint>> vpMP;
//     pMap->GetAllMapPoints(vpMP);
//     BundleAdjustment(vpKFs, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust, useSchurTrick);
// }


void BundleAdjustment(const std::vector<std::shared_ptr<Frame>> &vpKFs, const std::vector<std::shared_ptr<MapPoint>> &vpMP, 
                        int nIterations, bool *pbStopFlag, const bool bRobust, const bool useSchurTrick,
                        const size_t maxKfIdatCand, const size_t minActkfid,const size_t maxMPIdatCand)
{
     if(vpKFs.size() < 2  || vpMP.size() < 10)
        return;


    g2o::SparseOptimizer optimizer;
    g2o::OptimizationAlgorithmLevenberg *solver;
    if (useSchurTrick)
        solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolver_6_3>(g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>()));
    else
        solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolverX>(g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolverX::PoseMatrixType>>()));
    
    optimizer.setAlgorithm(solver);
    HSLAM::camParams *cam_params = new HSLAM::camParams(vpKFs[0]->HCalib->fxl(), vpKFs[0]->HCalib->fyl(), vpKFs[0]->HCalib->cxl(), vpKFs[0]->HCalib->cyl());
    cam_params->setId(0);
    if (!optimizer.addParameter(cam_params))
    {
        std::cout << "could not set up BA calibration! do not run BA!" << std::endl;
        return;
    }

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // Set KeyFrame vertices
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        std::shared_ptr<Frame> pKF = vpKFs[i];
        if (pKF->isBad())
            continue;

        if (pKF->fs->KfId > maxKfIdatCand)
            continue;

        // SE3Vertex *vSE3 = new SE3Vertex();
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();


        auto Pose = pKF->fs->getPoseOpti(); 
        // vSE3->setEstimate(SE3(Pose.rotationMatrix(), Pose.translation()));

        // auto Pose = pKF->fs->getPoseInverse(); 
        vSE3->setEstimate(g2o::SE3Quat(Pose.rotationMatrix(), Pose.translation()));

        

        vSE3->setId(pKF->fs->KfId);

        if(pKF->fs->KfId >= minActkfid - 5)
            vSE3->setFixed(true);

        optimizer.addVertex(vSE3);
        if (pKF->fs->KfId > maxKFid)
            maxKFid = pKF->fs->KfId;
    }

    const float thHuber2D = sqrt(5.991); //1.345

    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());
    // std::vector<HSLAM::EdgeProjectinvDepth *> edges;

    std::vector<HSLAM::EdgeProjectPSI2UV *> edges;

    edges.reserve(vpMP.size());

    vector<double> vInfoOnce;
    vInfoOnce.reserve(vpMP.size());
    vector<double> vInformation;
    vInformation.reserve(4*vpMP.size());

    // Set MapPoint vertices
    for (size_t i = 0; i < vpMP.size(); i++)
    {
        std::shared_ptr<MapPoint> pMP = vpMP[i];
        if (pMP->isBad() || pMP->sourceFrame->isBad())
            continue;

        if (pMP->sourceFrame->fs->KfId > maxKfIdatCand || pMP->id > maxMPIdatCand) //if mapPoint created after the optimization is initiated!
            continue;

        // VertexPointIDepth *vPoint = new VertexPointIDepth(); 
        // vPoint->setUV(pMP->pt.cast<double>());
        // auto sourcePose = pMP->sourceFrame->fs->getPoseOptiInv();
        // vPoint->setEstimate(pMP->getidepth() / sourcePose.scale());

        HSLAM::VertexSBAPointXYZ *vPoint = new HSLAM::VertexSBAPointXYZ();
        auto sourcePose = pMP->sourceFrame->fs->getPoseOptiInv();
        Vec3 point_anchor = (cam_params->camInvMap(pMP->pt.cast<double>()) * (sourcePose.scale() / pMP->getidepth())); //pMP->sourceFrame->fs->getPose() *   // s
        vPoint->setEstimate(HSLAM::invert_depth(point_anchor));

        const int id = pMP->id + maxKFid + 1;
        vPoint->setId(id);

        if (useSchurTrick)
            vPoint->setMarginalized(true);

        if( pMP->sourceFrame->fs->KfId >= minActkfid - 5) //this KF was active when the candidate was detected need to fix its pose!
            vPoint->setFixed(true);

        optimizer.addVertex(vPoint);

        const std::map<std::shared_ptr<Frame>, size_t> observations = pMP->GetObservations();

        auto vanchor = dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertices().find(pMP->sourceFrame->fs->KfId)->second);
        int nEdges = 0;

        double Info = sqrtf(pMP->getidepthHessian());
        bool addedInfo = false;
        //SET EDGES
        for (std::map<std::shared_ptr<Frame>, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++)
        {
            std::shared_ptr<Frame> pKF = mit->first;
            if (pKF->isBad() || pKF->fs->KfId > maxKfIdatCand || pKF == pMP->sourceFrame)
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeys[mit->second];

            Vec2 obs((double)kpUn.pt.x, (double)kpUn.pt.y);

            // HSLAM::EdgeProjectinvDepth *e = new HSLAM::EdgeProjectinvDepth();
            HSLAM::EdgeProjectPSI2UV *e = new HSLAM::EdgeProjectPSI2UV();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vPoint));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertices().find(pKF->fs->KfId)->second));
            e->setVertex(2, vanchor);

            e->setMeasurement(obs);
            if(!addedInfo)
            {
                addedInfo = true;
                vInfoOnce.push_back(Info);
            }
            // const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
            vInformation.push_back(Info);
            // e->setInformation(Eigen::Matrix2d::Identity()); // * pMP->getVariance();

            if (bRobust)
            {
                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuber2D);
            }

            e->setParameterId(0, 0);
            edges.push_back(e);
            optimizer.addEdge(e);
        }

        if (nEdges == 0) //|| pMP->getDirStatus() == MapPoint::active
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i] = true;
        }
        else
        {
            vbNotIncludedMP[i] = false;
        }
    }


    double stdDev = getStdDev(vInfoOnce);
    for (int i = 0, iend = edges.size(); i < iend; ++i)
        edges[i]->setInformation(Eigen::Matrix2d::Identity() * (vInformation[i] / (stdDev + 0.00001)));  //* invSigma2); //set this to take into account depth variance!


    optimizer.setVerbose(true);

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    //  Keyframes
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        std::shared_ptr<Frame> pKF = vpKFs[i];
        if (pKF->isBad() || pKF->fs->KfId > maxKfIdatCand) //pKF->getState() == Frame::active ||
            continue;

        // if (std::find(activeKfs.begin(), activeKfs.end(), pKF) != activeKfs.end()) //this KF was active when the candidate was detected need to fix its pose!
        //     continue;
        
        // SE3Vertex *vSE3 = static_cast<SE3Vertex *>(optimizer.vertex(pKF->fs->KfId));
        VertexSE3Expmap *vSE3 = static_cast<VertexSE3Expmap *>(optimizer.vertex(pKF->fs->KfId));
        
        if(vSE3->fixed())
            continue;
        double scale = pKF->fs->getPoseOpti().scale();
        // auto PoseSim3 = Sim3(vSE3->estimate().matrix());
        auto PoseSim3 = Sim3(SE3(vSE3->estimate().rotation().toRotationMatrix(), vSE3->estimate().translation()).matrix());

        PoseSim3.setScale(scale);
        pKF->fs->setPoseOpti(PoseSim3);
      
    }

    //Points
    for (size_t i = 0; i < vpMP.size(); i++)
    {
        if (vbNotIncludedMP[i])
            continue;

        std::shared_ptr<MapPoint> pMP = vpMP[i];

        if (pMP->isBad() || pMP->sourceFrame->isBad() || pMP->getDirStatus() != MapPoint::marginalized)
            continue;
        if (pMP->sourceFrame->fs->KfId > maxKfIdatCand || pMP->id > maxMPIdatCand) //if mapPoint created after the optimization is initiated!
            continue;


        // VertexPointIDepth *vPoint = static_cast<VertexPointIDepth *>(optimizer.vertex(pMP->id + maxKFid + 1));
        VertexSBAPointXYZ *vPoint = static_cast<VertexSBAPointXYZ *>(optimizer.vertex(pMP->id + maxKFid + 1));

        if(vPoint->fixed())
            continue;
        auto scale = pMP->sourceFrame->fs->getPoseOptiInv().scale();
        // pMP->updateDepthfromInd((float)vPoint->estimate()/ scale); //vPoint->estimate()(2)

        pMP->updateDepthfromInd((float)vPoint->estimate()[2]*scale); //vPoint->estimate()(2)
        pMP->updateGlobalPose();
        pMP->UpdateNormalAndDepth();
    }
}

// void BundleAdjustment(const std::vector<std::shared_ptr<Frame>> &vpKFs, const std::vector<std::shared_ptr<MapPoint>> &vpMP, 
//                         std::vector<std::shared_ptr<Frame>> &activeKfs, std::vector<std::shared_ptr<MapPoint>> &activeMps,
//                                  int nIterations, bool *pbStopFlag, const bool bRobust, const bool useSchurTrick,
//                                  int totalKfId, int currMaxKF, int currMaxMp)
// {
//     if(vpKFs.size() < 2  || vpMP.size() < 10)
//         return;
    
//     vector<bool> vbNotIncludedMP;
//     vbNotIncludedMP.resize(vpMP.size());

//     g2o::SparseOptimizer optimizer;
//     g2o::OptimizationAlgorithmLevenberg *solver;
//     if (useSchurTrick)
//         solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolver_6_3>(g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>()));
//     else
//         solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolverX>(g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolverX::PoseMatrixType>>()));
    
//     optimizer.setAlgorithm(solver);

 
//     camParams *cam_params = new camParams(vpKFs[0]->HCalib->fxl(), vpKFs[0]->HCalib->fyl(), vpKFs[0]->HCalib->cxl(), vpKFs[0]->HCalib->cyl());
//     cam_params->setId(0);
//     if (!optimizer.addParameter(cam_params))
//     {
//         std::cout << "could not set up BA calibration! do not run BA!" << std::endl;
//         return;
//     }

//     if (pbStopFlag)
//         optimizer.setForceStopFlag(pbStopFlag);

//     long unsigned int maxKFid = 0;

//     // Set KeyFrame vertices
//     for (size_t i = 0; i < vpKFs.size(); i++)
//     {
//         std::shared_ptr<Frame> pKF = vpKFs[i];
//         if (pKF->isBad())
//             continue;

//         if(pKF->fs->KfId > currMaxKF)
//             continue;

//         g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();

//         auto Pose = pKF->fs->getPoseOpti(); //
//         Pose.setScale(1.0);
//         vSE3->setEstimate(g2o::SE3Quat(Pose.rotationMatrix(), Pose.translation())); //this removes the scale from the pose, remember to re-add it later

//         vSE3->setId(pKF->fs->KfId);

//         if(std::find(activeKfs.begin(), activeKfs.end(), pKF) != activeKfs.end()) //this KF was active when the candidate was detected need to fix its pose!
//             vSE3->setFixed(true);
        
//         optimizer.addVertex(vSE3);
//         if (pKF->fs->KfId > maxKFid)
//             maxKFid = pKF->fs->KfId;
//     }

//     const float thHuber2D = sqrt(5.99);
    
//     // Set MapPoint vertices
//     for (size_t i = 0; i < vpMP.size(); i++)
//     {
//         std::shared_ptr<MapPoint> pMP = vpMP[i];
//         if (pMP->isBad() || pMP->sourceFrame->isBad() )
//             continue;

//         if (pMP->sourceFrame->fs->KfId > currMaxKF || pMP->id > currMaxMp)
//             continue;

        

//         VertexPointDepth *vPoint = new VertexPointDepth(); //g2o::vertexSBAPointXYZ..
//         vPoint->setUV(pMP->pt.cast<double>());
        
//         auto sourcePose = pMP->sourceFrame->fs->getPoseOpti();
//         // sourcePose.setScale(1.0); //map points are already converted to the true scale, no need to rescale
//         // vPoint->setEstimate(invert_depth(sourcePose * pMP->getWorldPose().cast<double>())); //MP in frame
//         // auto ptinFrame = (sourcePose * pMP->getWorldPose().cast<double>());
//         vPoint->setEstimate(pMP->getidepth() / sourcePose.scale());

//         const int id = pMP->id + maxKFid + 1;
//         vPoint->setId(id);

//         if(useSchurTrick)
//             vPoint->setMarginalized(true);

        
//         // if(pMP->getDirStatus() == MapPoint::active || pMP->sourceFrame->getState()==Frame::active || pMP->id > currMaxMp)
//         //     vPoint->setFixed(true);
        
//         // if(pMP->sourceFrame->fs->KfId > currMaxKF)
//         //     vPoint->setFixed(true);
        
//         if(std::find(activeKfs.begin(), activeKfs.end(), pMP->sourceFrame) != activeKfs.end()) //this KF was active when the candidate was detected need to fix its pose!
//             vPoint->setFixed(true);
        
        

//         optimizer.addVertex(vPoint);

//         const std::map<std::shared_ptr<Frame>, size_t> observations = pMP->GetObservations();
        
//         auto vanchor = dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertices().find(pMP->sourceFrame->fs->KfId)->second);
//         int nEdges = 0;
        
//         //SET EDGES
//         for (std::map<std::shared_ptr<Frame>, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++)
//         {

//             std::shared_ptr<Frame> pKF = mit->first;
//             if (pKF->isBad() || pKF->fs->KfId > currMaxKF)
//                 continue;

//             nEdges++;

//             const cv::KeyPoint &kpUn = pKF->mvKeys[mit->second];

//             Eigen::Matrix<double, 2, 1> obs;
//             obs << kpUn.pt.x, kpUn.pt.y;


//             EdgeProjectinvDepth *e = new EdgeProjectinvDepth();
//             e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vPoint));
//             e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertices().find(pKF->fs->KfId)->second));
//             e->setVertex(2, vanchor);

            
//             e->setMeasurement(obs);
//             // const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
//             e->setInformation(Eigen::Matrix2d::Identity()); // * pMP->getVariance();
           

//             if (bRobust)
//             {
//                 g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
//                 e->setRobustKernel(rk);
//                 rk->setDelta(thHuber2D);
//             }

//             e->setParameterId(0, 0);
//             optimizer.addEdge(e);
//         }

//         if (nEdges == 0 ) //|| pMP->getDirStatus() == MapPoint::active
//         {
//             optimizer.removeVertex(vPoint);
//             vbNotIncludedMP[i] = true;
//         }
//         else
//         {
//             vbNotIncludedMP[i] = false;
//         }
//     }
//             optimizer.setVerbose(true);

//         // Optimize!
//         optimizer.initializeOptimization();
//     optimizer.optimize(nIterations);

//     // Recover optimized data

//     //Keyframes
//     for (size_t i = 0; i < vpKFs.size(); i++)
//     {
//         std::shared_ptr<Frame> pKF = vpKFs[i];
//         if (pKF->isBad() || pKF->fs->KfId > currMaxKF) //pKF->getState() == Frame::active || 
//             continue;
        
//         if(std::find(activeKfs.begin(), activeKfs.end(),pKF) != activeKfs.end()) //this KF was active when the candidate was detected need to fix its pose!
//             continue;

//         g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKF->fs->KfId));
//         g2o::SE3Quat SE3quat = vSE3->estimate();
//         // if (nLoopKF == 0)
//         // {
       
//         auto PoseSim3 = Sim3(SE3(SE3quat.rotation().toRotationMatrix(), SE3quat.translation()).matrix());
//         PoseSim3.setScale(pKF->fs->getPoseOpti().scale());
//         pKF->fs->setPoseOpti(PoseSim3);
//         // }
//         // else
//         // {
//         //     pKF->mTcwGBA.create(4, 4, CV_32F);
//         //     Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
//         //     pKF->mnBAGlobalForKF = nLoopKF;
//         // }
//     }

//     //Points
//     for (size_t i = 0; i < vpMP.size(); i++)
//     {
//         if (vbNotIncludedMP[i])
//             continue;

//         std::shared_ptr<MapPoint> pMP = vpMP[i];

//         if (pMP->isBad() || pMP->sourceFrame->isBad() ||  pMP->getDirStatus() != MapPoint::marginalized)
//             continue;
//         if (pMP->id > currMaxMp)
//             continue;
//         if(std::find(activeKfs.begin(), activeKfs.end(), pMP->sourceFrame) != activeKfs.end()) //this KF was active when the candidate was detected need to fix its pose!
//             continue;

//         VertexPointDepth *vPoint = static_cast<VertexPointDepth *>(optimizer.vertex(pMP->id + maxKFid + 1));

//         // if (nLoopKF == 0)
//         // {
//             // auto scale = pMP->sourceFrame->fs->getPoseOpti().scale();
//             pMP->updateDepthfromInd((float)vPoint->estimate()); //vPoint->estimate()(2)
//             pMP->updateGlobalPose();
//             // pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
//             pMP->UpdateNormalAndDepth();
//         // }
//         // else
//         // {
//         //     pMP->mPosGBA.create(3, 1, CV_32F);
//         //     Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
//         //     pMP->mnBAGlobalForKF = nLoopKF;
//         // }
//     }
// }

} // namespace HSLAM