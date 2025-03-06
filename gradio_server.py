import os
import time
try:
    import triton
except ImportError:
    pass
from pathlib import Path
from loguru import logger
from datetime import datetime
import gradio as gr
import random
import json
from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.constants import NEGATIVE_PROMPT
from hyvideo.modules.attenion import get_attention_modes
from hyvideo.modules.models import get_linear_split_map
import asyncio
from mmgp import offload, safetensors2, profile_type 
import torch
import gc
import traceback

attention_modes_supported = get_attention_modes()

args = parse_args()
args.flow_reverse = True


lock_ui_attention = False
lock_ui_transformer = False
lock_ui_compile = False


force_profile_no = int(args.profile)
verbose_level = int(args.verbose)
preload =int(args.preload)

quantizeTransformer = args.quantize_transformer

transformer_choices_t2v=["ckpts/hunyuan-video-t2v-720p/transformers/hunyuan_video_720_bf16.safetensors", "ckpts/hunyuan-video-t2v-720p/transformers/hunyuan_video_720_quanto_int8.safetensors", "ckpts/hunyuan-video-t2v-720p/transformers/fast_hunyuan_video_720_quanto_int8.safetensors"]
transformer_choices_i2v=["ckpts/hunyuan-video-i2v-720p/transformers/hunyuan_video_i2v_720_bf16.safetensors", "ckpts/hunyuan-video-i2v-720p/transformers/hunyuan_video_i2v_720_quanto_int8.safetensors", "ckpts/hunyuan-video-t2v-720p/transformers/fast_hunyuan_video_720_quanto_int8.safetensors"]
text_encoder_choices = ["ckpts/text_encoder/llava-llama-3-8b-v1_1_vlm_fp16.safetensors", "ckpts/text_encoder/llava-llama-3-8b-v1_1_vlm_quanto_int8.safetensors"]

server_config_filename = "gradio_config.json"

if not Path(server_config_filename).is_file():
    server_config = {"attention_mode" : "auto",  
                     "transformer_filename": transformer_choices_t2v[1], 
                     "transformer_filename_i2v": transformer_choices_i2v[1],
                     "text_encoder_filename" : text_encoder_choices[1],
                     "compile" : "",
                     "default_ui": "t2v",
                     "vae_config": 0,
                     "profile" : profile_type.LowRAM_LowVRAM }

    with open(server_config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(server_config))
else:
    with open(server_config_filename, "r", encoding="utf-8") as reader:
        text = reader.read()
    server_config = json.loads(text)


transformer_filename_t2v = server_config["transformer_filename"]
transformer_filename_i2v = server_config.get("transformer_filename_i2v", transformer_choices_i2v[1]) ########
if transformer_filename_i2v == transformer_choices_t2v[1]:
    transformer_filename_i2v = transformer_choices_i2v[1]
text_encoder_filename = server_config["text_encoder_filename"]
if not "vlm" in text_encoder_filename:
     text_encoder_filename = text_encoder_filename.replace("v1_1_","v1_1_vlm_")
attention_mode = server_config["attention_mode"]

if len(args.attention)> 0:
    if args.attention in ["auto", "sdpa", "sage", "sage2", "flash", "xformers"]:
        attention_mode = args.attention
        lock_ui_attention = True
    else:
        raise Exception(f"Unknown attention mode '{args.attention}'")

profile =  force_profile_no if force_profile_no >=0 else server_config["profile"]
compile = server_config.get("compile", "")
vae_config = server_config.get("vae_config", 0)
if len(args.vae_config) > 0:
    vae_config = int(args.vae_config)

default_ui = server_config.get("default_ui", "t2v") 
use_image2video = default_ui != "t2v"
if args.t2v:
    use_image2video = False
if args.i2v:
    use_image2video = True

args.i2v_mode = use_image2video
if use_image2video:
    args.model = "HYVideo-T/2"    
    lora_dir =args.lora_dir_i2v
    lora_preselected_preset = args.lora_preset_i2v
else:
    args.model = "HYVideo-T/2-cfgdistill"
    lora_dir =args.lora_dir
    lora_preselected_preset = args.lora_preset

default_tea_cache = 0
if args.fast or args.fastest:
    transformer_filename_t2v = transformer_choices_t2v[2]
    attention_mode="sage2" if "sage2" in attention_modes_supported else "sage"
    default_tea_cache = 0.15
    lock_ui_attention = True
    lock_ui_transformer = True

if args.fastest or args.compile:
    compile="transformer"
    lock_ui_compile = True

fast_hunyan = "fast" in transformer_filename_t2v

#transformer_filename = "ckpts/hunyuan-video-t2v-720p/transformers/hunyuan_video_720_bf16.safetensors"
#transformer_filename = "ckpts/hunyuan-video-t2v-720p/transformers/hunyuan_video_720_quanto_int8.safetensors"
#transformer_filename = "ckpts/hunyuan-video-t2v-720p/transformers/fast_hunyuan_video_720_quanto_int8.safetensors"


# transformer_filename_i2v = "ckpts/hunyuan-video-i2v-720p/transformers/hunyuan_video_i2v_720_bf16.safetensors"
# transformer_filename_i2v = "ckpts/hunyuan-video-i2v-720p/transformers/hunyuan_video_i2v_720_quanto_int8.safetensors" 

#text_encoder_filename = "ckpts/text_encoder/llava-llama-3-8b-v1_1_fp16.safetensors"
#text_encoder_filename = "ckpts/text_encoder/llava-llama-3-8b-v1_1_quanto_int8.safetensors"


#attention_mode="sage"
#attention_mode="sage2"
#attention_mode="flash"
#attention_mode="sdpa"
#attention_mode="xformers"
# compile = "transformer"

def download_models(transformer_filename, text_encoder_filename):
    def computeList(filename):
        pos = filename.rfind("/")
        filename = filename[pos+1:]
        return [filename]        
    
    from huggingface_hub import hf_hub_download, snapshot_download    
    repoId = "DeepBeepMeep/HunyuanVideo" 
    if use_image2video:
        sourceFolderList = ["text_encoder_2", "text_encoder", "hunyuan-video-t2v-720p/vae", "hunyuan-video-i2v-720p/transformers" ]
    else:
        sourceFolderList = ["text_encoder_2", "text_encoder", "hunyuan-video-t2v-720p/vae", "hunyuan-video-t2v-720p/transformers" ]
    fileList = [ [], ["config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"] + computeList(text_encoder_filename) , [],  computeList(transformer_filename) ]
    targetRoot = "ckpts/" 
    for sourceFolder, files in zip(sourceFolderList,fileList ):
        if len(files)==0:
            if not Path(targetRoot + sourceFolder).exists():
                snapshot_download(repo_id=repoId,  allow_patterns=sourceFolder +"/*", local_dir= targetRoot)
        else:
             for onefile in files:      
                if not os.path.isfile(targetRoot + sourceFolder + "/" + onefile ):          
                    hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = targetRoot, subfolder=sourceFolder)


offload.default_verboseLevel = verbose_level

download_models(transformer_filename_i2v if use_image2video else transformer_filename_t2v, text_encoder_filename) 

def sanitize_file_name(file_name):
    return file_name.replace("/","").replace("\\","").replace(":","").replace("|","").replace("?","").replace("<","").replace(">","").replace("\"","") 
def preprocess_loras(sd):
    if not use_image2video:
        return sd
    new_sd = {}
    for k,v in sd.items():
        repl_list = ["double_blocks", "single_blocks", "final_layer", "img_mlp", "img_attn_qkv", "img_attn_proj","img_mod", "txt_mlp", "txt_attn_qkv","txt_attn_proj", "txt_mod", "linear1", 
                     "linear2", "modulation",  "mlp_fc1"]
        src_list = [k +"_" for k in repl_list] +  ["_" + k for k in repl_list]
        tgt_list = [k +"." for k in repl_list] +  ["." + k for k in repl_list]

        if k.startswith("Hunyuan_video_I2V_lora_"):
            k = k.replace("Hunyuan_video_I2V_lora_","diffusion_model.")
            k = k.replace("lora_up","lora_B")
            k = k.replace("lora_down","lora_A")
            if "txt_in_individual" in k:
                pass
            for s,t in zip(src_list, tgt_list):
                k = k.replace(s,t)
            if  "individual_token_refiner" in k:
                k = k.replace("txt_in_individual_token_refiner_blocks_", "txt_in.individual_token_refiner.blocks.")
                k = k.replace("_mlp_fc", ".mlp.fc",)
                k = k.replace(".mlp_fc", ".mlp.fc",)
        new_sd[k] = v
    return new_sd

def extract_preset(lset_name, loras):
    lset_name = sanitize_file_name(lset_name)
    if not lset_name.endswith(".lset"):
        lset_name_filename = os.path.join(lora_dir, lset_name + ".lset" ) 
    else:
        lset_name_filename = os.path.join(lora_dir, lset_name ) 

    if not os.path.isfile(lset_name_filename):
        raise gr.Error(f"Preset '{lset_name}' not found ")

    with open(lset_name_filename, "r", encoding="utf-8") as reader:
        text = reader.read()
    lset = json.loads(text)

    loras_choices_files = lset["loras"]
    loras_choices = []
    missing_loras = []
    for lora_file in loras_choices_files:
        loras_choice_no = loras.index(os.path.join(lora_dir, lora_file))
        if loras_choice_no < 0:
            missing_loras.append(lora_file)
        else:
            loras_choices.append(str(loras_choice_no))

    if len(missing_loras) > 0:
        raise gr.Error(f"Unable to apply Lora preset '{lset_name} because the following Loras files are missing: {missing_loras}")
    
    loras_mult_choices = lset["loras_mult"]
    return loras_choices, loras_mult_choices

def setup_loras(pipe,  lora_dir, lora_preselected_preset, split_linear_modules_map = None):
    # lora_weight =["ckpts/arny_lora.safetensors"] # 'ohwx person' ,; 'wick'
    # lora_multi = [1.0]
    loras =[]
    loras_names = []
    default_loras_choices = []
    default_loras_multis_str = ""
    loras_presets = []

    from pathlib import Path

    if lora_dir != None:
        if not os.path.isdir(lora_dir):
            raise Exception("--lora-dir should be a path to a directory that contains Loras")

    default_lora_preset = ""

    if lora_dir != None:
        import glob
        dir_loras =  glob.glob( os.path.join(lora_dir , "*.sft") ) + glob.glob( os.path.join(lora_dir , "*.safetensors") ) 
        dir_loras.sort()
        loras += [element for element in dir_loras if element not in loras ]

        dir_presets =  glob.glob( os.path.join(lora_dir , "*.lset") ) 
        dir_presets.sort()
        loras_presets = [ Path(Path(file_path).parts[-1]).stem for file_path in dir_presets]

    if len(loras) > 0:
        loras_names = [ Path(lora).stem for lora in loras  ]
        offload.load_loras_into_model(pipe.transformer, loras,  activate_all_loras=False, split_linear_modules_map = split_linear_modules_map, preprocess_sd= preprocess_loras) #lora_multiplier,

    if len(lora_preselected_preset) > 0:
        if not os.path.isfile(os.path.join(lora_dir, lora_preselected_preset + ".lset")):
            raise Exception(f"Unknown preset '{lora_preselected_preset}'")
        default_lora_preset = lora_preselected_preset
        default_loras_choices, default_loras_multis_str= extract_preset(default_lora_preset, loras)

    return loras, loras_names, default_loras_choices, default_loras_multis_str, default_lora_preset, loras_presets


def load_models(i2v,  lora_dir,  lora_preselected_preset ):
    download_models(transformer_filename_i2v if i2v else transformer_filename_t2v, text_encoder_filename) 

    if False:
        from magic_141_video.infer_ti2v import init_magic_141_video
        hunyuan_video_sampler = init_magic_141_video()
        pipe = { "transformer" : hunyuan_video_sampler.model, "text_encoder_2" : hunyuan_video_sampler.text_encoder_2, "vae" : hunyuan_video_sampler.vae  }
        pipe.update(offload.extract_models(hunyuan_video_sampler.text_encoder_vlm), "text_encoder")
    else:
        hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(transformer_filename_i2v if i2v else transformer_filename_t2v, text_encoder_filename, attention_mode = attention_mode, args=args,  device="cpu") #pinToMemory = pinToMemory, partialPinning = partialPinning,  
        pipe = hunyuan_video_sampler.pipeline
        pipe.transformer.any_compilation = len(compile)>0

    kwargs = { "extraModelsToQuantize": None}

    if profile == 2 or profile == 4:
        kwargs["budgets"] = { "transformer" : 100 if preload  == 0 else preload, "text_encoder" : 100, "*" : 1000 }
    elif profile == 3:
        kwargs["budgets"] = { "*" : "70%" }

    split_linear_modules_map = get_linear_split_map()
    offload.split_linear_modules(pipe.transformer, split_linear_modules_map )
    loras, loras_names, default_loras_choices, default_loras_multis_str, default_lora_preset, loras_presets = setup_loras(pipe,  lora_dir, lora_preselected_preset, split_linear_modules_map)
    offloadobj = offload.profile(pipe, profile_no= profile, compile = compile, quantizeTransformer = quantizeTransformer, **kwargs)  


    return hunyuan_video_sampler, offloadobj, loras, loras_names, default_loras_choices, default_loras_multis_str, default_lora_preset, loras_presets

hunyuan_video_sampler, offloadobj,  loras, loras_names, default_loras_choices, default_loras_multis_str, default_lora_preset, loras_presets = load_models(use_image2video, lora_dir, lora_preselected_preset )
gen_in_progress = False

def get_auto_attention():
    for attn in ["sage2","sage","sdpa"]:
        if attn in attention_modes_supported:
            return attn
    return "sdpa"

def get_default_steps_flow(fast_hunyan ):
    if use_image2video:
        return 30, 17 
    else:
        return 6 if fast_hunyan else 30, 17.0 if fast_hunyan else 7.0 

def generate_header(fast_hunyan, compile, attention_mode):
    header = "<H2 ALIGN=CENTER><SPAN> ----------------- "
    header += ("Fast HunyuanVideo model" if fast_hunyan else "HunyuanVideo model") 
    header += (" Image to Video" if use_image2video else " Text to Video") 
    header += " (attention mode: " + (attention_mode if attention_mode!="auto" else "auto/" + get_auto_attention() )
    if attention_mode not in attention_modes_supported:
        header += " -NOT INSTALLED-"

    if compile:
        header += ", pytorch compilation ON"
    header += ") -----------------</SPAN></H2>"

    return header

def apply_changes(  state,
                    transformer_t2v_choice,
                    transformer_i2v_choice,
                    text_encoder_choice,
                    attention_choice,
                    compile_choice,
                    profile_choice,
                    vae_config_choice,
                    default_ui_choice ="t2v",
):

    if args.lock_config:
        return

    if gen_in_progress:
        yield "<DIV ALIGN=CENTER>Unable to change config when a generation is in progress</DIV>"
        return
    global offloadobj, hunyuan_video_sampler, loras, loras_names, default_loras_choices, default_loras_multis_str, default_lora_preset, loras_presets
    server_config = {"attention_mode" : attention_choice,  
                     "transformer_filename": transformer_choices_t2v[transformer_t2v_choice], 
                     "transformer_filename_i2v": transformer_choices_i2v[transformer_i2v_choice],  ##########
                     "text_encoder_filename" : text_encoder_choices[text_encoder_choice],
                     "compile" : compile_choice,
                     "profile" : profile_choice,
                     "vae_config" : vae_config_choice,
                     "default_ui" : default_ui_choice,
                       }

    if Path(server_config_filename).is_file():
        with open(server_config_filename, "r", encoding="utf-8") as reader:
            text = reader.read()
        old_server_config = json.loads(text)
        if lock_ui_transformer:
            server_config["transformer_filename"] = old_server_config["transformer_filename"]
            server_config["transformer_filename_i2v"] = old_server_config["transformer_filename_i2v"]
        if lock_ui_attention:
            server_config["attention_mode"] = old_server_config["attention_mode"]
        if lock_ui_compile:
            server_config["compile"] = old_server_config["compile"]

    with open(server_config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(server_config))

    changes = []
    for k, v in server_config.items():
        v_old = old_server_config.get(k, None)
        if v != v_old:
            changes.append(k)

    state["config_changes"] = changes
    state["config_new"] = server_config
    state["config_old"] = old_server_config

    global attention_mode, profile, compile, transformer_filename_t2v, transformer_filename_i2v, text_encoder_filename, vae_config
    attention_mode = server_config["attention_mode"]
    profile = server_config["profile"]
    compile = server_config["compile"]
    transformer_filename_t2v = server_config["transformer_filename"]
    transformer_filename_i2v = server_config["transformer_filename_i2v"]
    text_encoder_filename = server_config["text_encoder_filename"]
    vae_config = server_config["vae_config"]

    if  all(change in ["attention_mode", "vae_config"] for change in changes ):
        if "attention_mode" in changes:
            pass

    else:
        hunyuan_video_sampler = None
        offloadobj.release()
        offloadobj = None
        yield "<DIV ALIGN=CENTER>Please wait while the new configuration is being applied</DIV>"

        hunyuan_video_sampler, offloadobj,  loras, loras_names, default_loras_choices, default_loras_multis_str, default_lora_preset, loras_presets = load_models(use_image2video, lora_dir,  lora_preselected_preset )


    yield "<DIV ALIGN=CENTER>The new configuration has been succesfully applied</DIV>"

    # return "<DIV ALIGN=CENTER>New Config file created. Please restart the Gradio Server</DIV>"

def update_defaults(state, num_inference_steps,flow_shift):
    if "config_changes" not in state:
        return get_default_steps_flow(False)
    changes = state["config_changes"] 
    server_config = state["config_new"] 
    old_server_config = state["config_old"] 

    new_fast_hunyuan = "fast" in server_config["transformer_filename"]
    old_fast_hunyuan = "fast" in old_server_config["transformer_filename"]

    if  "transformer_filename" in changes:
        if new_fast_hunyuan != old_fast_hunyuan:
            num_inference_steps, flow_shift = get_default_steps_flow(new_fast_hunyuan)

    header = generate_header(new_fast_hunyuan, server_config["compile"], server_config["attention_mode"] )
    return num_inference_steps, flow_shift, header 


from moviepy.editor import ImageSequenceClip
import numpy as np

def save_video(final_frames, output_path, fps=24):
    assert final_frames.ndim == 4 and final_frames.shape[3] == 3, f"invalid shape: {final_frames} (need t h w c)"
    if final_frames.dtype != np.uint8:
        final_frames = (final_frames * 255).astype(np.uint8)
    ImageSequenceClip(list(final_frames), fps=fps).write_videofile(output_path, verbose= False, logger = None)

def build_callback(state, pipe, progress, status, num_inference_steps):
    def callback(step_idx, t, latents):
        step_idx += 1         
        if state.get("abort", False):
            # pipe._interrupt = True
            status_msg = status + " - Aborting"    
        elif step_idx  == num_inference_steps:
            status_msg = status + " - VAE Decoding"    
        else:
            status_msg = status + " - Denoising"   

        progress( (step_idx , num_inference_steps) , status_msg  ,  num_inference_steps)
            
    return callback

def abort_generation(state):
    if "in_progress" in state:
        state["abort"] = True
        hunyuan_video_sampler.pipeline._interrupt= True
        return gr.Button(interactive=  False)
    else:
        return gr.Button(interactive=  True)

def refresh_gallery(state):
    file_list = state.get("file_list", None)      
    return file_list
        
def finalize_gallery(state):
    choice = 0
    if "in_progress" in state:
        del state["in_progress"]
        choice = state.get("selected",0)
    
    time.sleep(0.2)
    gen_in_progress = False
    return gr.Gallery(selected_index=choice), gr.Button(interactive=  True)

def select_video(state , event_data: gr.EventData):
    data=  event_data._data
    if data!=None:
        state["selected"] = data.get("index",0)
    return 

def expand_slist(slist, num_inference_steps ):
    new_slist= []
    inc =  len(slist) / num_inference_steps 
    pos = 0
    for i in range(num_inference_steps):
        new_slist.append(slist[ int(pos)])
        pos += inc
    return new_slist


def generate_video(
    prompt,
    negative_prompt,
    resolution,
    video_length,
    seed,
    num_inference_steps,
    guidance_scale,
    flow_shift,
    embedded_guidance_scale,
    repeat_generation,
    tea_cache,
    loras_choices,
    loras_mult_choices,
    image_to_continue,
    video_to_continue,
    max_frames,
    RIFLEx_setting,
    state,
    progress=gr.Progress() #track_tqdm= True

):
    
    from PIL import Image
    import numpy as np
    import tempfile


    if hunyuan_video_sampler == None:
        raise gr.Error("Unable to generate a Video while a new configuration is being applied.")
    if attention_mode == "auto":
        attn = get_auto_attention()
    elif attention_mode in attention_modes_supported:
        attn = attention_mode
    else:
        raise gr.Error(f"You have selected attention mode '{attention_mode}'. However it is not installed on your system. You should either install it or switch to the default 'sdpa' attention.")

    transformer = hunyuan_video_sampler.pipeline.transformer
    transformer.attention_mode = attn
    for module in transformer.double_blocks:
        module.attention_mode = attn
    for module in transformer.single_blocks:
        module.attention_mode = attn

    global gen_in_progress
    gen_in_progress = True
    temp_filename = None
    if len(prompt) ==0:
        return
    prompts = prompt.replace("\r", "").split("\n")
    if use_image2video:
        if image_to_continue is not None:
            if isinstance(image_to_continue, list):
                image_to_continue = [ tup[0] for tup in image_to_continue ]
            else:
                image_to_continue = [image_to_continue]
            if len(prompts) >= len(image_to_continue):
                if len(prompts) % len(image_to_continue) !=0:
                    raise gr.Error("If there are more text prompts than input images the number of text prompts should be dividable by the number of images")
                rep = len(prompts) // len(image_to_continue)
                new_image_to_continue = []
                for i, _ in enumerate(prompts):
                    new_image_to_continue.append(image_to_continue[i//rep] )
                image_to_continue = new_image_to_continue 
            else: 
                if len(image_to_continue) % len(prompts)  !=0:
                    raise gr.Error("If there are more input images than text prompts the number of images should be dividable by the number of text prompts")
                rep = len(image_to_continue) // len(prompts)  
                new_prompts = []
                for i, _ in enumerate(image_to_continue):
                    new_prompts.append(  prompts[ i//rep] )
                prompts = new_prompts
        elif video_to_continue != None and len(video_to_continue) >0 :
            input_image_or_video_path = video_to_continue
            # pipeline.num_input_frames = max_frames
            # pipeline.max_frames = max_frames
        else:
            return
    else:
        input_image_or_video_path = None


    if len(loras) > 0:
        def is_float(element: any) -> bool:
            if element is None: 
                return False
            try:
                float(element)
                return True
            except ValueError:
                return False
        list_mult_choices_nums = []
        if len(loras_mult_choices) > 0:
            list_mult_choices_str = loras_mult_choices.split(" ")
            for i, mult in enumerate(list_mult_choices_str):
                mult = mult.strip()
                if "," in mult:
                    multlist = mult.split(",")
                    slist = []
                    for smult in multlist:
                        if not is_float(smult):                
                            raise gr.Error(f"Lora sub value no {i+1} ({smult}) in Multiplier definition '{multlist}' is invalid")
                        slist.append(float(smult))
                    slist = expand_slist(slist, num_inference_steps )
                    list_mult_choices_nums.append(slist)
                else:
                    if not is_float(mult):                
                        raise gr.Error(f"Lora Multiplier no {i+1} ({mult}) is invalid")
                    list_mult_choices_nums.append(float(mult))
        if len(list_mult_choices_nums ) < len(loras_choices):
            list_mult_choices_nums  += [1.0] * ( len(loras_choices) - len(list_mult_choices_nums ) )

        offload.activate_loras(hunyuan_video_sampler.pipeline.transformer, loras_choices, list_mult_choices_nums)

    seed = None if seed == -1 else seed
    if use_image2video:
        width, height = 0, 0
    else:
        width, height = resolution.split("x")
        width, height = int(width), int(height)
    if not use_image2video:
        negative_prompt = "" # not applicable in the inference

    if "abort" in state:
        del state["abort"]
    state["in_progress"] = True
    state["selected"] = 0
 
    enable_riflex = RIFLEx_setting == 0 and video_length > (5* 24) or RIFLEx_setting == 1
    # VAE Tiling
    device_mem_capacity = torch.cuda.get_device_properties(0).total_memory / 1048576
    vae = hunyuan_video_sampler.vae
    if vae_config == 0:
        if device_mem_capacity >= 24000:
            use_vae_config = 1            
        elif device_mem_capacity >= 16000:
            use_vae_config = 3          
        elif device_mem_capacity >= 12000:
            use_vae_config = 4
        else:
            use_vae_config = 5
    else:
        use_vae_config = vae_config

    if use_vae_config == 1:
        sample_tsize = 32
        sample_size = 256  
    elif use_vae_config == 2:
        sample_tsize = 64
        sample_size = 192  
    elif use_vae_config == 3:
        sample_tsize = 32
        sample_size = 192  
    elif use_vae_config == 4:
        sample_tsize = 16
        sample_size = 256  
    else:
        sample_tsize = 16
        sample_size = 192  

    vae.tile_sample_min_tsize = sample_tsize
    vae.tile_latent_min_tsize = sample_tsize // vae.time_compression_ratio
    vae.tile_sample_min_size = sample_size
    vae.tile_latent_min_size = int(sample_size / (2 ** (len(vae.config.block_out_channels) - 1)))
    vae.tile_overlap_factor = 0.25

   # TeaCache   
    trans = hunyuan_video_sampler.pipeline.transformer
    trans.enable_teacache = tea_cache > 0
 
    import random
    if seed == None or seed <0:
        seed = random.randint(0, 999999999)

    file_list = []
    state["file_list"] = file_list    
    from einops import rearrange
    save_path = os.path.join(os.getcwd(), "gradio_outputs")
    os.makedirs(save_path, exist_ok=True)
    video_no = 0
    total_video =  repeat_generation * len(prompts)
    abort = False
    start_time = time.time()
    for prompt in prompts:
        for _ in range(repeat_generation):
            if abort:
                break

            if trans.enable_teacache:
                trans.num_steps = num_inference_steps
                trans.cnt = 0
                trans.rel_l1_thresh = tea_cache #0.15 # 0.1 for 1.6x speedup, 0.15 for 2.1x speedup
                trans.accumulated_rel_l1_distance = 0
                trans.previous_modulated_input = None
                trans.previous_residual = None

            video_no += 1
            status = f"Video {video_no}/{total_video}"
            progress(0, desc=status + " - Encoding Prompt" )   
            
            callback = build_callback(state, hunyuan_video_sampler.pipeline, progress, status, num_inference_steps)

            if use_image2video:
                outputs = hunyuan_video_sampler.predict_i2v(
                    prompt=prompt, 
                    video_length=(video_length // 4)* 4 + 1 ,
                    seed=seed,
                    negative_prompt=negative_prompt,
                    infer_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_videos_per_prompt=1,
                    flow_shift=flow_shift,
                    batch_size=1,
                    embedded_guidance_scale=embedded_guidance_scale,
                    i2v_mode=True,
                    i2v_resolution=resolution, #args.i2v_resolution, 720p , 360p #540p
                    i2v_image = image_to_continue[video_no-1],
                    callback = callback,
                    callback_steps = 1,
                    enable_riflex= enable_riflex

                )

                # input_image_or_video_path
                # raise Exception("image 2 video not yet supported") #################
            else:
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    outputs = hunyuan_video_sampler.predict_t2v(
                        prompt=prompt,
                        height=height,
                        width=width, 
                        video_length=(video_length // 4)* 4 + 1 ,
                        seed=seed,
                        negative_prompt=negative_prompt,
                        infer_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        num_videos_per_prompt=1,
                        flow_shift=flow_shift,
                        batch_size=1,
                        embedded_guidance_scale=embedded_guidance_scale,
                        callback = callback,
                        callback_steps = 1,
                        enable_riflex= enable_riflex

                    )
                except Exception as e:
                    gen_in_progress = False
                    if temp_filename!= None and  os.path.isfile(temp_filename):
                        os.remove(temp_filename)
                    offload.last_offload_obj.unload_all()
                    # if compile:
                    #     cache_size = torch._dynamo.config.cache_size_limit                                      
                    #     torch.compiler.reset()
                    #     torch._dynamo.config.cache_size_limit = cache_size
                    if trans.enable_teacache:
                        trans.previous_modulated_input = None
                        trans.previous_residual = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    s = str(e)
                    keyword_list = ["vram", "VRAM", "memory", "triton", "cuda", "allocat"]
                    VRAM_crash= False
                    if any( keyword in s for keyword in keyword_list):
                        VRAM_crash = True
                    else:
                        stack = traceback.extract_stack(f=None, limit=5)
                        for frame in stack:
                            if any( keyword in frame.name for keyword in keyword_list):
                                VRAM_crash = True
                                break
                    if VRAM_crash:
                        raise gr.Error("The generation of the video has encountered an error: it is likely that you have unsufficient VRAM and you should therefore reduce the video resolution or its number of frames.")
                    else:
                        raise gr.Error(f"The generation of the video has encountered an error, please check your terminal for more information. '{s}'")


            if trans.enable_teacache:
                trans.previous_modulated_input = None
                trans.previous_residual = None

            samples = outputs['samples']
            if samples != None:
                samples = samples.to("cpu")
            outputs['samples'] = None
            offload.last_offload_obj.unload_all()
            gc.collect()
            torch.cuda.empty_cache()

            if samples == None:
                end_time = time.time()
                abort = True
                yield f"Video generation was aborted. Total Generation Time: {end_time-start_time:.1f}s"
            else:
                idx = 0
                # just in case one day we will have enough VRAM for batch generation ...
                for i,sample in enumerate(samples):
                    # sample = samples[0]
                    video = rearrange(sample.cpu().numpy(), "c t h w -> t h w c")

                    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%Hh%Mm%Ss")
                    file_name = f"{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','').strip()}.mp4".replace(':',' ').replace('\\',' ')
                    idx = 0 
                    basis_video_path = os.path.join(os.getcwd(), "gradio_outputs", file_name)        
                    video_path = basis_video_path
                    while True:
                        if  not Path(video_path).is_file():
                            idx = 0
                            break
                        idx += 1
                        video_path = basis_video_path[:-4] + f"_{idx}" + ".mp4"


                    save_video(video, video_path )
                    print(f"New video saved to Path: "+video_path)
                    file_list.append(video_path)
                    if video_no < total_video:
                        yield  status
                    else:
                        end_time = time.time()
                        yield f"Total Generation Time: {end_time-start_time:.1f}s"
            seed += 1
  
    if temp_filename!= None and  os.path.isfile(temp_filename):
        os.remove(temp_filename)
    gen_in_progress = False

new_preset_msg = "Enter a Name for a Lora Preset or Choose One Above"

def save_lset(lset_name, loras_choices, loras_mult_choices):
    global loras_presets
    
    if len(lset_name) == 0 or lset_name== new_preset_msg:
        gr.Info("Please enter a name for the preset")
        lset_choices =[("Please enter a name for a Lora Preset","")]
    else:
        lset_name = sanitize_file_name(lset_name)

        loras_choices_files = [ Path(loras[int(choice_no)]).parts[-1] for choice_no in loras_choices  ]
        lset  = {"loras" : loras_choices_files, "loras_mult" : loras_mult_choices}
        lset_name_filename = lset_name + ".lset" 
        full_lset_name_filename = os.path.join(lora_dir, lset_name_filename) 

        with open(full_lset_name_filename, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(lset))

        if lset_name in loras_presets:
            gr.Info(f"Lora Preset '{lset_name}' has been updated")
        else:
            gr.Info(f"Lora Preset '{lset_name}' has been created")
            loras_presets.append(Path(Path(lset_name_filename).parts[-1]).stem )
        lset_choices = [ ( preset, preset) for preset in loras_presets ]
        lset_choices.append( (new_preset_msg, ""))

    return gr.Dropdown(choices=lset_choices, value= lset_name)

def delete_lset(lset_name):
    global loras_presets
    lset_name_filename = os.path.join(lora_dir,  sanitize_file_name(lset_name) + ".lset" )
    if len(lset_name) > 0 and lset_name != new_preset_msg:
        if not os.path.isfile(lset_name_filename):
            raise gr.Error(f"Preset '{lset_name}' not found ")
        os.remove(lset_name_filename)
        pos = loras_presets.index(lset_name) 
        gr.Info(f"Lora Preset '{lset_name}' has been deleted")
        loras_presets.remove(lset_name)
    else:
        pos = len(loras_presets) 
        gr.Info(f"Choose a Preset to delete")

    lset_choices = [ (preset, preset) for preset in loras_presets]
    lset_choices.append((new_preset_msg, ""))
    return  gr.Dropdown(choices=lset_choices, value= lset_choices[pos][1])

def apply_lset(lset_name, loras_choices, loras_mult_choices):

    if len(lset_name) == 0 or lset_name== new_preset_msg:
        gr.Info("Please choose a preset in the list or create one")
    else:
        loras_choices, loras_mult_choices= extract_preset(lset_name, loras)
        gr.Info(f"Lora Preset '{lset_name}' has been applied")

    return loras_choices, loras_mult_choices

def create_demo():
    
    default_inference_steps, default_flow_shift = get_default_steps_flow(fast_hunyan)
    
    with gr.Blocks() as demo:
        state = gr.State({})
       
        if use_image2video:
            gr.Markdown("<div align=center><H1>HunyuanVideo<SUP>GP</SUP> v6 - AI Image To Video Generator (<A HREF='https://github.com/deepbeepmeep/HunyuanVideoGP'>Updates</A> / <A HREF='https://github.com/Tencent/HunyuanVideo'>Original by Tencent</A>)</H1></div>")
        else:
            gr.Markdown("<div align=center><H1>HunyuanVideo<SUP>GP</SUP> v6 - AI Text To Video Generator (<A HREF='https://github.com/deepbeepmeep/HunyuanVideoGP'>Updates</A> / <A HREF='https://github.com/Tencent/HunyuanVideo'>Original by Tencent</A>)</H1></div>")

        gr.Markdown("<FONT SIZE=3>With this new release by <B>DeepBeepMeep</B>,  VRAM consumption has been divided 3 and you can now generate 12s of a 1280 * 720 video + Loras with 24 GB of VRAM at no quality loss</I></FONT>")

        if use_image2video and False:
            pass
        else:
            gr.Markdown("The resolution and the duration of the video will depend on the amount of VRAM your GPU has, for instance if you have 24 GB of VRAM (RTX 3090 / RTX 4090), the limits are as follows:")
            gr.Markdown("- 848 x 480: 261 frames (10.5s) / 385 frames (16s) with Pytorch compilation (please note there is no point going beyond 10.5s duration as the videos will look redundant)")
            gr.Markdown("- 1280 x 720: 192 frames (8s) / 261 frames (10.5s) with Pytorch compilation")
        gr.Markdown("In order to find the sweet spot you will need try different resolution / duration and reduce these if the app is hanging : in the very worst case one generation step should not take more than 2 minutes. If it is the case you may be running out of RAM / VRAM.")
        gr.Markdown("Please note that if your turn on compilation, the first generation step of the first video generation will be slow due to the compilation. Therefore all your tests should be done with compilation turned off.")


        # css = """<STYLE>
        #         h2 { width: 100%;  text-align: center; border-bottom: 1px solid #000; line-height: 0.1em; margin: 10px 0 20px;  } 
        #         h2 span {background:#fff;  padding:0 10px; }</STYLE>"""
        # gr.HTML(css)

        header = gr.Markdown(generate_header(fast_hunyan, compile, attention_mode) , visible= not args.lock_config )            

        with gr.Accordion("Video Engine Configuration - click here to change it", open = False):
            gr.Markdown("For the changes to be effective you will need to restart the gradio_server. Some choices below may be locked if the app has been launched by specifying a config preset.")

            with gr.Column():
                index = transformer_choices_t2v.index(transformer_filename_t2v)
                index = 0 if index ==0 else index
                transformer_t2v_choice = gr.Dropdown(
                    choices=[
                        ("Hunyuan Text to Video 16 bits - the default engine in its original glory, offers a slightly better image quality but slower and requires more RAM", 0),
                        ("Hunyuan Text to Video quantized to 8 bits (recommended) - the default engine but quantized", 1),
                        ("Fast Hunyuan Text to Video quantized to 8 bits - requires less than 10 steps but worse quality", 2), 
                    ],
                    value= index,
                    label="Transformer model for Text to Video",
                    interactive= not lock_ui_transformer,
                    visible= not use_image2video
                 )

                index = transformer_choices_i2v.index(transformer_filename_i2v)
                index = 0 if index ==0 else index
                transformer_i2v_choice = gr.Dropdown(
                    choices=[
                        ("Hunyuan Image to Video 16 bits - the default engine in its original glory, offers a slightly better image quality but slower and requires more RAM", 0),
                        ("Hunyuan Image to Video quantized to 8 bits (recommended) - the default engine but quantized", 1),
                        # ("Fast Hunyuan Video quantized to 8 bits - requires less than 10 steps but worse quality", 2), 
                    ],
                    value= index,
                    label="Transformer model for Image to Video",
                    interactive= not lock_ui_transformer,
                    visible = use_image2video
                 )

                index = text_encoder_choices.index(text_encoder_filename)
                index = 0 if index ==0 else index

                text_encoder_choice = gr.Dropdown(
                    choices=[
                        ("Llava Llama 1.1 16 bits - unquantized text encoder, better quality uses more RAM", 0),
                        ("Llava Llama 1.1 quantized to 8 bits - quantized text encoder, slightly worse quality but uses less RAM", 1),
                    ],
                    value= index,
                    label="Text Encoder model"
                 )
                def check(mode): 
                    if not mode in attention_modes_supported:
                        return " (NOT INSTALLED)"
                    else:
                        return ""
                attention_choice = gr.Dropdown(
                    choices=[
                        ("Auto : pick sage2 > sage > sdpa depending on what is installed", "auto"),
                        ("Scale Dot Product Attention: default, always available", "sdpa"),
                        ("Flash" + check("flash")+ ": good quality - requires additional install (usually complex to set up on Windows without WSL)", "flash"),
                        ("Xformers" + check("xformers")+ ": good quality - requires additional install (usually complex, may consume less VRAM to set up on Windows without WSL)", "xformers"),
                        ("Sage" + check("sage")+ ": 30% faster but slightly worse quality - requires additional install (usually complex to set up on Windows without WSL)", "sage"),
                        ("Sage2" + check("sage2")+ ": 40% faster but slightly worse quality - requires additional install (usually complex to set up on Windows without WSL)", "sage2"),
                    ],
                    value= attention_mode,
                    label="Attention Type",
                    interactive= not lock_ui_attention
                 )
                gr.Markdown("Beware: when restarting the server or changing a resolution or video duration, the first step of generation for a duration / resolution may last a few minutes due to recompilation")
                compile_choice = gr.Dropdown(
                    choices=[
                        ("ON: works only on Linux / WSL", "transformer"),
                        ("OFF: no other choice if you have Windows without using WSL", "" ),
                    ],
                    value= compile,
                    label="Compile Transformer (up to 50% faster and 30% more frames but requires Linux / WSL and Flash or Sage attention)",
                    interactive= not lock_ui_compile
                 )              


                vae_config_choice = gr.Dropdown(
                    choices=[
                ("Auto", 0),
                ("32 frames * 256 px * 256 px (recommended 24+ GB VRAM)", 1),
                ("64 frames * 192 px * 192 px (recommended 24+ GB VRAM)", 2),
                ("32 frames * 192 px * 192 px (recommended 16+ GB VRAM)", 3),
                ("16 frames * 256 px * 256 px (recommended 12+ GB VRAM)", 4),
                ("16 frames * 192 px * 192 px ", 5),
                    ],
                    value= vae_config,
                    label="VAE Tiling - reduce time of VAE decoding (if the last stage takes more than 2 minutes). The smaller the tile, the worse the quality. You may use larger tiles than recommended on shorter videos."
                 )

                profile_choice = gr.Dropdown(
                    choices=[
                ("HighRAM_HighVRAM, profile 1: at least 48 GB of RAM and 24 GB of VRAM, the fastest for short videos a RTX 3090 / RTX 4090", 1),
                ("HighRAM_LowVRAM, profile 2 (Recommended): at least 48 GB of RAM and 12 GB of VRAM, the most versatile profile with high RAM, better suited for RTX 3070/3080/4070/4080 or for RTX 3090 / RTX 4090 with large pictures batches or long videos", 2),
                ("LowRAM_HighVRAM, profile 3: at least 32 GB of RAM and 24 GB of VRAM, adapted for RTX 3090 / RTX 4090 with limited RAM for good speed short video",3),
                ("LowRAM_LowVRAM, profile 4 (Default): at least 32 GB of RAM and 12 GB of VRAM, if you have little VRAM or want to generate longer videos",4),
                ("VerylowRAM_LowVRAM, profile 5: (Fail safe): at least 16 GB of RAM and 10 GB of VRAM, if you don't have much it won't be fast but maybe it will work",5)
                    ],
                    value= profile,
                    label="Profile (for power users only, not needed to change it)"
                 )

                default_ui_choice = gr.Dropdown(
                    choices=[
                        ("Text to Video", "t2v"),
                        ("Image to Video", "i2v"),
                    ],
                    value= default_ui,
                    label="Default mode when launching the App if not '--t2v' ot '--i2v' switch is specified when launching the server ",
                    visible= False ############
                 )                

                msg = gr.Markdown()            
                apply_btn  = gr.Button("Apply Changes")


        with gr.Row():
            with gr.Column():
                video_to_continue = gr.Video(label= "Video to continue", visible= use_image2video and False) #######  
                if args.multiple_images:  
                    image_to_continue = gr.Gallery(
                            label="Images as a starting point for new videos", type ="pil", #file_types= "image", 
                            columns=[3], rows=[1], object_fit="contain", height="auto", selected_index=0, interactive= True, visible=use_image2video)
                else:
                    image_to_continue = gr.Image(label= "Image as a starting point for a new video",type ="pil", visible=use_image2video)

                if use_image2video:
                    prompt = gr.Textbox(label="Prompts (multiple prompts separated by carriage returns will generate multiple videos)", value="Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field.", lines=3)
                else:
                    prompt = gr.Textbox(label="Prompts (multiple prompts separated by carriage returns will generate multiple videos)", value="A large orange octopus is seen resting on the bottom of the ocean floor, blending in with the sandy and rocky terrain. Its tentacles are spread out around its body, and its eyes are closed. The octopus is unaware of a king crab that is crawling towards it from behind a rock, its claws raised and ready to attack. The crab is brown and spiny, with long legs and antennae. The scene is captured from a wide angle, showing the vastness and depth of the ocean. The water is clear and blue, with rays of sunlight filtering through. The shot is sharp and crisp, with a high dynamic range. The octopus and the crab are in focus, while the background is slightly blurred, creating a depth of field effect.", lines=3)
                with gr.Row():

                    if use_image2video:
                        resolution = gr.Dropdown(
                            choices=[
                                ("720p", "720p"),
                                ("540p", "540p"),
                                ("360p", "360p"),
                            ],
                            value="540p",
                            label="Resolution (video will have the same height / width ratio than the original image)"
                        )
                    else:
                        resolution = gr.Dropdown(
                            choices=[
                                # 720p
                                ("1280x720 (16:9, 720p)", "1280x720"),
                                ("720x1280 (9:16, 720p)", "720x1280"), 
                                ("1104x832 (4:3, 720p)", "1104x832"),
                                ("832x1104 (3:4, 720p)", "832x1104"),
                                ("960x960 (1:1, 720p)", "960x960"),
                                # 540p
                                ("960x544 (16:9, 540p)", "960x544"),
                                ("848x480 (16:9, 540p)", "848x480"),
                                ("544x960 (9:16, 540p)", "544x960"),
                                ("832x624 (4:3, 540p)", "832x624"), 
                                ("624x832 (3:4, 540p)", "624x832"),
                                ("720x720 (1:1, 540p)", "720x720"),
                                # ("540x320 (1:1, 540p)", "540x320"),
                            ],
                            value="848x480",
                            label="Resolution"
                        )




                with gr.Row():
                    with gr.Column():
                        video_length = gr.Slider(5, 337, value=97, step=4, label="Number of frames (24 = 1s)")
                    with gr.Column():
                        num_inference_steps = gr.Slider(1, 100, value=  default_inference_steps, step=1, label="Number of Inference Steps")

                with gr.Row():
                    max_frames = gr.Slider(1, 100, value=9, step=1, label="Number of input frames to use for Video2World prediction", visible=use_image2video and False) #########
    

                with gr.Row(visible= len(loras)>0):
                    lset_choices = [ (preset, preset) for preset in loras_presets ] + [(new_preset_msg, "")]
                    with gr.Column(scale=5):
                        lset_name = gr.Dropdown(show_label=False, allow_custom_value= True, scale=5, filterable=False, choices= lset_choices, value=default_lora_preset)
                    with gr.Column(scale=1):
                        # with gr.Column():
                        with gr.Row(height=17):
                            apply_lset_btn = gr.Button("Apply Lora Preset", size="sm", min_width= 1)
                        with gr.Row(height=17):
                            save_lset_btn = gr.Button("Save", size="sm", min_width= 1)
                            delete_lset_btn = gr.Button("Delete", size="sm", min_width= 1)


                loras_choices = gr.Dropdown(
                    choices=[
                        (lora_name, str(i) ) for i, lora_name in enumerate(loras_names)
                    ],
                    value= default_loras_choices,
                    multiselect= True,
                    visible= len(loras)>0,
                    label="Activated Loras"
                )
                loras_mult_choices = gr.Textbox(label="Loras Multipliers (1.0 by default) separated by space characters or carriage returns", value=default_loras_multis_str, visible= len(loras)>0 )

                show_advanced = gr.Checkbox(label="Show Advanced Options", value=False)
                with gr.Row(visible=False) as advanced_row:
                    with gr.Column():
                        seed = gr.Slider(-1, 999999999, value=-1, step=1, label="Seed (-1 for random)") 
                        repeat_generation = gr.Slider(1, 25.0, value=1.0, step=1, label="Number of Generated Video per prompt") 
                        with gr.Row():
                            negative_prompt = gr.Textbox(label="Negative Prompt", value="")
                        with gr.Row():
                            guidance_scale = gr.Slider(1.0, 20.0, value=1.0, step=0.5, label="Guidance Scale", visible= use_image2video)
                            embedded_guidance_scale = gr.Slider(1.0, 20.0, value=6.0, step=0.5, label="Embedded Guidance Scale", visible= not use_image2video)
                            flow_shift = gr.Slider(0.0, 25.0, value= default_flow_shift, step=0.1, label="Flow Shift") 
                        tea_cache_setting = gr.Dropdown(
                            choices=[
                                ("Disabled", 0),
                                ("Fast (x1.6 speed up)", 0.1), 
                                ("Faster (x2.1 speed up)", 0.15), 
                            ],
                            value=default_tea_cache,
                            label="Tea Cache acceleration (the faster the acceleration the higher the degradation of the quality of the video. Consumes VRAM)"
                        )

                        RIFLEx_setting = gr.Dropdown(
                            choices=[
                                ("Auto (ON if Video longer than 5s)", 0),
                                ("Always ON", 1), 
                                ("Always OFF", 2), 
                            ],
                            value=0,
                            label="RIFLex positional embedding to generate long video"
                        )

                show_advanced.change(fn=lambda x: gr.Row(visible=x), inputs=[show_advanced], outputs=[advanced_row])
            
            with gr.Column():
                gen_status = gr.Text(label="Status", interactive= False) 
                output = gr.Gallery(
                        label="Generated videos", show_label=False, elem_id="gallery"
                    , columns=[3], rows=[1], object_fit="contain", height="auto", selected_index=0, interactive= False)
                generate_btn = gr.Button("Generate")
                abort_btn = gr.Button("Abort")

        save_lset_btn.click(save_lset, inputs=[lset_name, loras_choices, loras_mult_choices], outputs=[lset_name])
        delete_lset_btn.click(delete_lset, inputs=[lset_name], outputs=[lset_name])
        apply_lset_btn.click(apply_lset, inputs=[lset_name,loras_choices, loras_mult_choices], outputs=[loras_choices, loras_mult_choices])

        gen_status.change(refresh_gallery, inputs = [state], outputs = output )

        abort_btn.click(abort_generation,state,abort_btn )
        output.select(select_video, state, None )

        generate_btn.click(
            fn=generate_video,
            inputs=[
                prompt,
                negative_prompt,
                resolution,
                video_length,
                seed,
                num_inference_steps,
                guidance_scale,
                flow_shift,
                embedded_guidance_scale,
                repeat_generation,
                tea_cache_setting,
                loras_choices,
                loras_mult_choices,
                image_to_continue,
                video_to_continue,
                max_frames,
                RIFLEx_setting,
                state
            ],
            outputs= [gen_status] #,state 

        ).then( 
            finalize_gallery,
            [state], 
            [output , abort_btn]
        )

        apply_btn.click(
                fn=apply_changes,
                inputs=[
                    state,
                    transformer_t2v_choice,
                    transformer_i2v_choice,
                    text_encoder_choice,
                    attention_choice,
                    compile_choice,                            
                    profile_choice,
                    vae_config_choice,
                    default_ui_choice,
                ],
                outputs= msg
            ).then( 
            update_defaults, 
            [state, num_inference_steps,  flow_shift], 
            [num_inference_steps,  flow_shift, header]
                )

    return demo

if __name__ == "__main__":
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    server_port = int(args.server_port)

    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if server_port == 0:
        server_port = int(os.getenv("SERVER_PORT", "7860"))

    server_name = args.server_name
    if len(server_name) == 0:
        server_name = os.getenv("SERVER_NAME", "localhost")

        
    demo = create_demo()
    if args.open_browser:
        import webbrowser 
        if server_name.startswith("http"):
            url = server_name 
        else:
            url = "http://" + server_name 
        webbrowser.open(url + ":" + str(server_port), new = 0, autoraise = True)

    demo.launch(server_name=server_name, server_port=server_port, share=args.share)

 
