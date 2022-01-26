if(BUILD_TESTING)
    # In order to have header files show properly
    # in VS, they have to be put into a source group
    # and in the add_executable() call

    set(SOURCE_LIST
        Basic.cpp
        Kernel.cu)

    set(HEADER_LIST
        Basic.h
        ${PROJECT_SOURCE_DIR}/src/Vortigaunt.h
        ${PROJECT_SOURCE_DIR}/src/Detail.h
        ${PROJECT_SOURCE_DIR}/src/Utility.h)

    source_group("Source Files" FILES ${SOURCE_LIST})
    source_group("Header Files" FILES ${HEADER_LIST})

    add_executable(TestBasicBin
                   ${HEADER_LIST}
                   ${SOURCE_LIST}
                   )

    target_include_directories(TestBasicBin PRIVATE
                               ${PROJECT_SOURCE_DIR}/src)

    set_target_properties(TestBasicBin PROPERTIES FOLDER "test/basic")

    add_test(NAME TestBasic
             COMMAND TestBasicBin)
endif()