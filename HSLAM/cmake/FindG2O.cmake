# Find the header files

Set(G2O_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/Thirdparty/CompiledLibs/include/g2o)

# Macro to unify finding both the debug and release versions of the
# libraries; this is adapted from the OpenSceneGraph FIND_LIBRARY
# macro.

macro(FIND_G2O_LIBRARY MYLIBRARY MYLIBRARYNAME)

  find_library("${MYLIBRARY}_DEBUG"
    NAMES "g2o_${MYLIBRARYNAME}_d"
    PATHS
    ${PROJECT_SOURCE_DIR}/Thirdparty/CompiledLibs/lib
    )
  
  find_library(${MYLIBRARY}
    NAMES "g2o_${MYLIBRARYNAME}"
    PATHS
    ${PROJECT_SOURCE_DIR}/Thirdparty/CompiledLibs/lib
    )

  
  if(NOT ${MYLIBRARY}_DEBUG)
    if(MYLIBRARY)
      set(${MYLIBRARY}_DEBUG ${MYLIBRARY})
    endif(MYLIBRARY)
  endif( NOT ${MYLIBRARY}_DEBUG)
  
endmacro(FIND_G2O_LIBRARY LIBRARY LIBRARYNAME)

# Find the core elements
FIND_G2O_LIBRARY(G2O_STUFF_LIBRARY stuff)
FIND_G2O_LIBRARY(G2O_CORE_LIBRARY core)

# Find the CLI library
FIND_G2O_LIBRARY(G2O_CLI_LIBRARY cli)

# Find the pluggable solvers
FIND_G2O_LIBRARY(G2O_SOLVER_CHOLMOD solver_cholmod)
FIND_G2O_LIBRARY(G2O_SOLVER_CSPARSE solver_csparse)
FIND_G2O_LIBRARY(G2O_SOLVER_CSPARSE_EXTENSION csparse_extension)
FIND_G2O_LIBRARY(G2O_SOLVER_DENSE solver_dense)
FIND_G2O_LIBRARY(G2O_SOLVER_PCG solver_pcg)
FIND_G2O_LIBRARY(G2O_SOLVER_SLAM2D_LINEAR solver_slam2d_linear)
FIND_G2O_LIBRARY(G2O_SOLVER_STRUCTURE_ONLY solver_structure_only)
FIND_G2O_LIBRARY(G2O_SOLVER_EIGEN solver_eigen)

# Find the predefined types
FIND_G2O_LIBRARY(G2O_TYPES_DATA types_data)
FIND_G2O_LIBRARY(G2O_TYPES_ICP types_icp)
FIND_G2O_LIBRARY(G2O_TYPES_SBA types_sba)
FIND_G2O_LIBRARY(G2O_TYPES_SCLAM2D types_sclam2d)
FIND_G2O_LIBRARY(G2O_TYPES_SIM3 types_sim3)
FIND_G2O_LIBRARY(G2O_TYPES_SLAM2D types_slam2d)
FIND_G2O_LIBRARY(G2O_TYPES_SLAM3D types_slam3d)

# G2O solvers declared found if we found at least one solver
set(G2O_SOLVERS_FOUND "NO")
if(G2O_SOLVER_CHOLMOD OR G2O_SOLVER_CSPARSE OR G2O_SOLVER_DENSE OR G2O_SOLVER_PCG OR G2O_SOLVER_SLAM2D_LINEAR OR G2O_SOLVER_STRUCTURE_ONLY OR G2O_SOLVER_EIGEN)
  set(G2O_SOLVERS_FOUND "YES")
endif(G2O_SOLVER_CHOLMOD OR G2O_SOLVER_CSPARSE OR G2O_SOLVER_DENSE OR G2O_SOLVER_PCG OR G2O_SOLVER_SLAM2D_LINEAR OR G2O_SOLVER_STRUCTURE_ONLY OR G2O_SOLVER_EIGEN)

# G2O itself declared found if we found the core libraries and at least one solver
set(G2O_FOUND "NO")
if(G2O_STUFF_LIBRARY AND G2O_CORE_LIBRARY AND G2O_INCLUDE_DIR AND G2O_SOLVERS_FOUND)
  set(G2O_FOUND "YES")
endif(G2O_STUFF_LIBRARY AND G2O_CORE_LIBRARY AND G2O_INCLUDE_DIR AND G2O_SOLVERS_FOUND)

if(G2O_FOUND)
  SET(G2O_LIBS cxsparse g2o_stuff g2o_core g2o_solver_cholmod g2o_solver_csparse g2o_csparse_extension g2o_solver_dense g2o_solver_pcg g2o_solver_slam2d_linear g2o_solver_structure_only g2o_solver_eigen g2o_types_data g2o_types_icp g2o_types_sba g2o_types_sclam2d g2o_types_sim3 g2o_types_slam2d g2o_types_slam3d) 
endif()
