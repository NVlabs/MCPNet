python extract_prototypes.py --case_name AWA2_test2 --device 1 --model ResNet --basic_model resnet50_relu --concept_per_layer 8 16 32 64 --cha 32 32 32 32
python find_topk_response.py --case_name AWA2_test2 --model ResNet --basic_model resnet50_relu --concept_per_layer 8 16 32 64 --cha 32 32 32 32 --device 1 --eigen_topk 1
python find_topk_area.py --case_name AWA2_test2 --model ResNet --basic_model resnet50_relu --concept_per_layer 8 16 32 64 --cha 32 32 32 32 --topk 5 --device 1 --eigen_topk 1 --masked --heatmap --individually
