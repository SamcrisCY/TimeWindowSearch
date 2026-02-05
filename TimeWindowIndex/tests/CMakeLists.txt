

# add_executable(buildindex buildindex.cpp)

# add_executable(search_multi search_multi.cpp)
# # add_executable(test_conbined test_conbined.cpp)
# add_executable(test_indexMerge test_indexMerge.cpp)
# add_executable(test_Live_update test_Live_update.cpp)
# add_executable(test_update test_update.cpp)
# # add_executable(test_Dead_update test_Dead_update.cpp)
# # add_executable(test_Dead_search test_Dead_search.cpp)
# add_executable(PreFilter PreFilter.cpp)
# target_link_libraries(PreFilter PRIVATE utils)


add_executable(test_search test_search.cpp)
target_link_libraries(test_search PRIVATE utils)

# add_executable(active_ablation active_ablation.cpp)
# target_link_libraries(active_ablation PRIVATE utils)

# add_executable(historical_ablation historical_ablation.cpp)
# target_link_libraries(historical_ablation PRIVATE utils)




# target_link_libraries(test_DIGRA PRIVATE utils)

# target_include_directories(test_search PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../DIGRA)
# target_link_libraries(search_multi PRIVATE utils)
# target_link_libraries(test_conbined PRIVATE utils)
# target_link_libraries(test_indexMerge PRIVATE utils)
# target_link_libraries(test_Live_update PRIVATE utils)
# target_link_libraries(test_update PRIVATE utils)


# target_link_libraries(test_Dead_update PRIVATE utils)
# target_link_libraries(test_Dead_search PRIVATE utils)
