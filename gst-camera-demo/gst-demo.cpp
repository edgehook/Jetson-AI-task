#include <cstdlib>
#include <gst/gst.h>
#include <gst/gstinfo.h>
#include <gst/app/gstappsink.h>
#include <glib-unix.h>
#include <dlfcn.h>

#include <iostream>
#include <sstream>
#include <thread>

using namespace std;

#define USE(x) ((void)(x))

static GstPipeline *gst_pipeline = nullptr;
static string launch_string;   

static void appsink_eos(GstAppSink * appsink, gpointer user_data)
{
    printf("app sink receive eos\n");
//    g_main_loop_quit (hpipe->loop);
}

static GstFlowReturn new_buffer(GstAppSink *appsink, gpointer user_data)
{
    GstSample *sample = NULL;

    g_signal_emit_by_name (appsink, "pull-sample", &sample,NULL);

    if (sample)
    {
        GstBuffer *buffer = NULL;
        GstCaps   *caps   = NULL;
        GstMapInfo map    = {0};

        caps = gst_sample_get_caps (sample);
        if (!caps)
        {
            printf("could not get snapshot format\n");
        }
        gst_caps_get_structure (caps, 0);
        buffer = gst_sample_get_buffer (sample);
        gst_buffer_map (buffer, &map, GST_MAP_READ);

        printf("map.size = %lu\n", map.size);

        gst_buffer_unmap(buffer, &map);

        gst_sample_unref (sample);
    }
    else
    {
        g_print ("could not make snapshot\n");
    }

    return GST_FLOW_OK;
}

int main(int argc, char** argv) {
    USE(argc);
    USE(argv);

    gst_init (&argc, &argv);

    GMainLoop *main_loop;
    main_loop = g_main_loop_new (NULL, FALSE);
    ostringstream launch_stream;
    int w = 3840;
    int h = 2160;
    GstAppSinkCallbacks callbacks = {appsink_eos, NULL, new_buffer};

    launch_stream
    << "nvv4l2camerasrc device=/dev/video1 ! "
    << "video/x-raw(memory:NVMM), format=UYVY, width="<< w <<", height="<< h 
    << ", interlace-mode=progressive, framerate=30/1 ! " 
    << "nvvidconv ! "
    << "video/x-raw(memory:NVMM), format=NV12, width="<< w <<", height="<< h <<" ! "
    << "nv3dsink";

    launch_string = launch_stream.str();

    g_print("Using launch string: %s\n", launch_string.c_str());

    GError *error = nullptr;
    gst_pipeline  = (GstPipeline*) gst_parse_launch(launch_string.c_str(), &error);

    if (gst_pipeline == nullptr) {
        g_print( "Failed to parse launch: %s\n", error->message);
        return -1;
    }
    if(error) g_error_free(error);

    GstElement *appsink_ = gst_bin_get_by_name(GST_BIN(gst_pipeline), "mysink");
    gst_app_sink_set_callbacks (GST_APP_SINK(appsink_), &callbacks, NULL, NULL);

    gst_element_set_state((GstElement*)gst_pipeline, GST_STATE_PLAYING); 

    sleep(5);
    g_main_loop_run (main_loop);

    gst_element_set_state((GstElement*)gst_pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(gst_pipeline));
    g_main_loop_unref(main_loop);

    g_print("going to exit \n");
    return 0;
}
