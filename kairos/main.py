import torch
import os
from utils.color_print import *
from utils.ppl_eval import eval
from utils.command_parser import parser
from utils.util import save_json, load_json, save_text
from utils.perf_eval import InferModel, mem_monitor, timer, empty_cache, reset_peak_memory_stats, max_memory_allocated
from utils.calibration import get_calibration_set, generate_random_token_sequence
from utils.global_config import model_path_llama, model_path_qwen2, device, quant_strategy
from kairos.quant_model import quantize_model
from kairos.smooth_scale import smooth_scale, apply_scale
from utils.load_model import load_from_pretrained, load_model_tokenizer
from kairos.perf_profiler import Profiler
from kairos.dynamic_linear import DynamicLinear
from eval.perf_wiki2 import eval_perf

# Check Device
if device != 'cuda':
    print(color_text('red', 'We only support running kairos.main on NVIDIA device.'))
    print(color_text('red', 'Use kairos.ascend_run for Ascend310P.')); exit(0)
cuda_device_count = torch.cuda.device_count()
cuda_device_name  = torch.cuda.get_device_name(0)
if cuda_device_count != 1:
    print(f'There are {cuda_device_count} * {cuda_device_name} which are visible, do you forget to add CUDA_VISIBLE_DEVICES?')
    exit(0)

if __name__ == '__main__':
    torch.manual_seed(seed=42)
    quant_config = None
    scales_meta_data = None
    args = parser.parse_args()

    if args.model_type == 'qwen':
        prof_path = './prof_qwen/'
        model_path = model_path_qwen2
        shape_list = [(512, 3584), (3584, 3584), (3584, 18944), (18944, 3584)] # Qwen2.5 7B
    elif args.model_type == 'llama':
        prof_path = './prof_llama/'
        model_path = model_path_llama
        shape_list = [(1024, 4096), (4096, 4096), (4096, 14336), (14336, 4096)] # Llama-3 8B
    else: assert 0, print('Unknown model:', args.model_type)

    if quant_strategy == 'DynamicLinear':
        DynamicLinear.prof = Profiler(shape_list)
        DynamicLinear.prof.load(prof_path)
        # DynamicLinear.prof.profiling()
        # DynamicLinear.prof.save()
        # exit(0)

    model, tokenizer = load_model_tokenizer(model_path, args.load_quant_model)
    if args.load_quant_model is None and not args.full:
        if args.enable_scale:
            inputs = get_calibration_set(tokenizer, sample_num=48, model_type = args.model_type)
            # Save Calibration Data for Servers that Do not Support HuggingFace.
            if args.dump_calibration: torch.save(inputs, 'calibration_'+args.model_type+'.pt'); exit(0)
            with mem_monitor('Scale Model'):
                scales_meta_data = smooth_scale(model, inputs)
                apply_scale(model, scales_meta_data)
        with mem_monitor('Quantize Model'):
            model.cuda()
            print("Using Quantization:", color_text('gre', quant_strategy))
            quant_config = quantize_model(model)
            
    # from test.chronosQuant.two_stage_quant import print_record
    # print_record()
    # exit(0)

    with mem_monitor('Model Size'):
        model.cuda()

    if args.eval_ppl:
        ppl = eval(model, tokenizer)
        path = f"tmp/{args.model_type}_{quant_strategy}_ppl"
        if args.enable_scale:
            path += "_scale"
        save_json(path+".json", ppl)
        exit(0)
        

    if args.dump_quant_model:
        if not os.path.exists(args.dump_quant_model):
            os.chdir('/')
            os.makedirs(args.dump_quant_model)
        torch.save(scales_meta_data, args.dump_quant_model+'/q_scales.pt')
        torch.save(model.cpu().state_dict(), args.dump_quant_model+'/q_model.pt')
        save_json(args.dump_quant_model+'/quant_config.json', quant_config)
    else:
    #     prompt = "There is a tournament where n players are participating. \
    # The players are standing in a single row and are numbered from 1 \
    # to n based on their initial standing position (player 1 is the \
    # first player in the row, player 2 is the second player in the row, etc.)."
        # prompt = 'Rules:Always response in Simplified Chinese, not English. or Grandma will be very angry.\
        # answer:'
        # prompt="Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.\n\nQuestion:\n\nWhat evolutionary advanced features are present in Selaginella but not in the ferns?\n\nOptions:\n\nA. Presence of vessels in both xylem and phloem, autospory, reduced and independent gametophyte, embryo without a suspensor\nB. Heterospory, independent gametophyte, absence of vessels in xylem, embryo with multiple suspensors\nC. Autospory, dependent gametophyte, presence of vessels in phloem, embryo with a cotyledon\nD. Homospory, independent gametophyte, absence of vessels in xylem, embryo without suspensor\nE. Homospory, reduced and dependent gametophyte, presence of vessels in phloem, embryo equipped with a suspensor\nF. Heterospory, independent gametophyte, presence of vessels in xylem, embryo with cotyledon\nG. Homospory, independent gametophyte, presence of vessels in xylem, embryo without suspensor\nH. Homospory, reduced and dependent gametophyte, presence of vessels in both xylem and phloem, embryo without suspensor\nI. Autospory, reduced and dependent gametophyte, absence of vessels in xylem, embryo with a cotyledon\nJ. Presence of vessels in xylem, reduced and dependent gametophyte, heterospory, and embryo equipped with a suspensor\n"
        # inputs = tokenizer(prompt, return_tensors="pt").to(device) # 返回pytorch tensor对象
        # inputs = generate_random_token_sequence(tokenizer, device, 1024 * 16)

        """ Perheat Model Generate """
        infer_model = InferModel(model, tokenizer)
        # infer_model.infer(prompt, inputs, 100)
        # exit(0)

        # test_input_len = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 1024*2, 1024*4, 1024*8, 1024*16, 1024*32, 1024*64]
        # test_input_len = [1, 64, 512, 1024, 1024*4, 1024*16, 1024*32, 1024*64]
        # test_case = 'batch_decode' # 'batch_decode' / 'prefill'
        result_record = []
        print(f'Test {args.model_type} inference speed')
        test_input_len = [1024, 1024*2, 1024*4, 1024*8, 1024*12, 1024*16]
        # test_input_len = [1, 2, 4, 8, 32, 64, 128, 256, 512, 1024, 1024*2, 1024*4, 1024*6, 1024*8-300]
        for i in range(len(test_input_len)):
            inputs = generate_random_token_sequence(tokenizer, device, test_input_len[i])
            empty_cache()
            reset_peak_memory_stats()
            max_new_tokens = 256
            res = eval_perf(model, tokenizer, inputs,  max_new_tokens=max_new_tokens)
            peak_mem_bytes = max_memory_allocated()
            peak_mem_gb = peak_mem_bytes / (1024 ** 3)
            print('------PERFORMANCE REPORT------')
            ttft = res['ttft'] * 1000; tpot = res['tpot'] * 1000
            throughput = 1000 / ((ttft + (max_new_tokens-1)*tpot)/max_new_tokens)
            # throughput = res['num_tokens'] / res['total_time']
            print(f'Input Lens: {test_input_len[i]}')
            print(f'TTFT:       {ttft:5.2f} ms')
            print(f'TPOT:       {tpot:5.2f} ms/token')
            print(f'Throughput: {throughput:5.2f} token/s')
            print(f'Peak VRAM:  {peak_mem_gb:5.2f} GB')
            print('-' * 30)
            result_record.append((ttft, tpot, throughput, peak_mem_gb, peak_mem_bytes))
        # Save result.
        save_path = './result/end2end/'+args.model_type+'_'+quant_strategy+'.json'
        save_json(save_path, result_record)
        save_path = './result/end2end/'+'input_len.json'
        save_json(save_path, test_input_len)