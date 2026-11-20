#include <math.h>
#include <cstdint>
#include "hal.h"
#include "stdint.h"
#include "usb_cdc.h"
#include "lio_lcd.h"
#include "sine_model_data.h"
#include "string.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
//#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"
//#include "tensorflow/lite/version.h"

#define WIDTH 160
#define HEIGHT 80

int map_f(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

/// Begin Global Vars
namespace{
 float x_pos = 0;
 int currentPoint = 0;
 const tflite::Model *model = nullptr;
 tflite::MicroInterpreter* interpreter = nullptr;
 TfLiteTensor* input = nullptr;
 TfLiteTensor* output = nullptr;
 int inference_count = 0;

 constexpr int kTensorArenaSize = 2*1024;
 uint8_t tensor_arena[kTensorArenaSize];
}
/// End Global Vars


void setup(){
    model = tflite::GetModel(g_sine_model_data);
    static tflite::MicroMutableOpResolver<4> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddQuantize();
    resolver.AddDequantize();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize
    );
    interpreter = &static_interpreter;

    TfLiteStatus allocate_status = interpreter->AllocateTensors();

    input = interpreter->input(0);
    output = interpreter->output(0);

    inference_count = 0;
}

int main(void){
    setup();    

    lio_lcd_config_t config;
    config.invert = 0;
    config.width = WIDTH;
    config.height = HEIGHT;
    config.offset_x = 0;
    config.offset_y = 24;
    config.background = 0x0000;

    lio_lcd_init(config);
    lio_lcd_clear(0x0000);

    while(1){
        for(volatile int i = 0; i < 1000000; i++);
        
        

        input->data.f[0] = x_pos;
        TfLiteStatus invokeStatus = interpreter->Invoke();
        float y_val = output->data.f[0];
        
        if(currentPoint<WIDTH)
            lio_lcd_draw_pixel(currentPoint, map_f(y_val, -1,1,0,HEIGHT), 0xF0F0);

        if(currentPoint>10)
            lio_lcd_draw_pixel(currentPoint-10,map_f(sin(x_pos), -1,1,0,HEIGHT),0xFFFF);
    
        currentPoint++;
        if(currentPoint>=WIDTH+9){
            currentPoint=0;
            lio_lcd_clear(0x0);
        }

        x_pos += 0.1f;
        if(x_pos >= 3.14156*2.0){
            x_pos = 0;
        }
    }
    
}