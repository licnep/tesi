find_package(OpenCV REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${GOLD_CONFIG_HEADERS_DIR}
    ${GOLD_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIR}
)

# elenco dei file da compilare
add_library(gold_app_dummy dummy.cpp HOGVisualizer.cpp)

# elenco delle librerie richieste da questa applicazione
target_link_libraries( gold_app_dummy
    ${GOLD_LIBRARIES}
    ${Boost_SERIALIZATION_LIBRARY}
    ${Boost_SIGNALS_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_THREAD_LIBRARY}
    #${OpenCV_LIBRARIES}
    -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect
)

message ("AAAopenCVdir=${OpenCV_LIBS},includeDir=${OpenCV_INCLUDE_DIRS}")

# mette il binario nella cartella dei plugind di gold
message ("GOLD_PLUGINS_DIR=${GOLD_PLUGINS_DIR}")

# TODO: qui ci vuole una directory di output diversa se stiamo impacchettando o installando
set_target_properties(gold_app_dummy PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${GOLD_DEV_PLUGINS_DIR})
install( TARGETS gold_app_dummy DESTINATION ${GOLD_PLUGINS_DIR} COMPONENT RunTime)
