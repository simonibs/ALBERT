# import tensorflow as tf

# with tf.Session() as sess:
#     saver = tf.train.import_meta_graph('./data/out_model.ckpt-best.meta', clear_devices=True)

#     graph = tf.get_default_graph()
#     sess = tf.Session()
#     saver.restore(sess, "./data/out_model.ckpt-best")
#     input_graph_def = graph.as_graph_def()
#     output_graph_def = graph_util.convert_variables_to_constants(
#             sess, # The session
#             input_graph_def, # input_graph_def is useful for retrieving the nodes 
#             output_node_names=['output']
#     )
#     output_graph="./out/my_model.pb"
#     with tf.gfile.GFile(output_graph, "wb") as f:
#         f.write(output_graph_def.SerializeToString())


import tensorflow as tf

meta_path = 'data/model.ckpt-best.meta' # Your .meta file
output_node_names = ['output:0']    # Output nodes

with tf.Session() as sess:
    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,save_path=tf.train.latest_checkpoint('data/'))

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open('./output_graph.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())