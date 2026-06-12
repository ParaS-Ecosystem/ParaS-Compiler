
message(STATUS "Configuring driver ...")

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_executable(parascc 
	driver/driver.cpp
	driver/Executor.cpp
	driver/Preprocessor.cpp
	driver/ParaSConsumer.cpp
	driver/ParaSQueueHandler.cpp
	driver/ParaSFuncParameterHandler.cpp
	driver/ParaSVarDeclHandler.cpp
	driver/ParaSSubmitHandler.cpp 
	driver/ParaSParallelforHandler.cpp  
)
target_link_libraries(parascc
 	PRIVATE
 	clangTooling
 	clangBasic
 	clangASTMatchers
	clangRewrite
)
message(STATUS "${CMAKE_CURRENT_LIST_DIR} --------------------")
target_include_directories(parascc PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_INSTALL_PREFIX}/include/>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../include>
)

install (TARGETS parascc DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
message(STATUS "Configured driver !")

message(STATUS "Configuring pruntime systems ...")

add_library(DeviceDiscoveryModule SHARED
	pruntime_system/device.cpp
	pruntime_system/gpu_detection.cpp
)

target_link_libraries(DeviceDiscoveryModule PRIVATE "" dl)
target_include_directories(DeviceDiscoveryModule PUBLIC 
	$<BUILD_INTERFACE:${PARAS_INCLUDE_DIR}>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../include>
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
)
install (TARGETS DeviceDiscoveryModule EXPORT install_exports LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/)

message(STATUS "Configured pruntime systems !")


