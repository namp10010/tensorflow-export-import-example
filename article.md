# Training TensorFlow models in Python and serving with Go

![img](https://miro.medium.com/max/700/1*qGsNiyUwc1ezOXYQ2MnCRA.jpeg)

Photo by [David Clode](https://unsplash.com/photos/d0CasEMHDQs?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

[travel audience](https://travelaudience.com/) is a digital advertising platform and as a data-driven company, we apply machine learning in several cases: selecting a creative we want to show to a particular user,or deciding if we want to participate in an auction are only two of many applications.

One of our more important models is used for click through rate (CTR) prediction, since the likelihood of a click on an advertising banner is an important ingredient for many decisions.

To be able to cope with the real-time requirements of programmatic advertising, our real-time system is implemented in [Go](https://golang.org/). However, for us Data Scientists, Python is the programming language we usually use to setup, train and evaluate our machine learning models. And this makes perfect sense, as Python became the go-to language for machine learning in recent years.

In our current stack, all model development and training happens outside of the real-time system and is implemented in Python. The parameters and coefficients of the trained models are exported periodically and then imported in Go to be used in our real-time system.

While this setup served us well in the past, it has several drawbacks:

1. some parts, like feature pre-processing and prediction, need to be implemented and maintained twice: first in Python and then in Go.
2. any change in the algorithms or our feature engineering requires re-implementing huge parts.
3. long feedback cycles when live testing new models, as many parties are involved.
4. Inconsistencies between Go & Python implementations are hard to detect and can decrease the performance

Overall, we were stuck with the current algorithms and we moved slower than we would have liked.

To overcome these issues, we started looking at [TensorFlow](https://www.tensorflow.org/), which is a great, open-source machine learning framework developed by Google. TensorFlow is famous for neural networks and deep learning, however it also supports simpler models like logistic regression. Most importantly, it has an implementation of the “Follow the regularized leader” optimizer, which is what we currently use for CTR prediction. Besides being an incredible machine learning library, TensorFlow has APIs for different programming languages, with Python and Go among them, bingo!

Unfortunately, TensorFlow’s Go API isn’t documented as well as the Python version and there are no tutorials that matched our use case. So here is some code that demonstrates how to first setup and train a model in Python and in the second part, how to import that model in Go and start predicting.

# Model setup and training with Python

Let’s create a basic Logistic Regression model to predict the likelihood of a click on an advertising banner. To keep it simple, we will only use three input features: the domain where the ad is shown, the device type (desktop vs mobile vs tablet) and the hour of the day. The first step is setting up our input feature columns. (Make sure to check [this excellent introduction](https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html) to TensorFlow’s feature columns.)

```
domain = tf.feature_column.categorical_column_with_hash_bucket(
 “domain”, 100000, dtype=tf.string)
hour = tf.feature_column.categorical_column_with_identity(“hour”, 24)
device_type = tf.feature_column.categorical_column_with_vocabulary_list(
 “device_type”, vocabulary_list=[“desktop”, “mobile”, “tablet”],
 default_value=0)feature_columns = [domain, hour, device_type]
```

Now we can setup our actual model using TensorFlow’s high-level `Estimator `interface. We are using the [Follow The Regularized Leader](https://research.google.com/pubs/pub41159.html) (FTRL) optimizer, which is an algorithm that provides adaptive learning rates and is particularly well-suited for large data sets with extremely high dimensionality and multi-level categorical features. It was developed with click-prediction in mind and perfectly fits our use case.

```
ftrl = tf.train.FtrlOptimizer(
    learning_rate=0.1,
    learning_rate_power=-0.5,
    l1_regularization_strength=0.001,
    l2_regularization_strength=0.0
    )model_dir = "."
estimator = tf.estimator.LinearClassifier(
    feature_columns=feature_columns,
    optimizer=ftrl,
    model_dir=model_dir
    )
```

To be able to train and evaluate our classifier, we need to provide an input function that defines how to load data and pipe it to the model. Because we are dealing with huge datasets, we cannot just read everything into memory but need to stream the data. Here is an input function that works with CSV files, it accepts a list of file paths and allows to iterate over the samples. The data files contain the raw input feature plus our target value `is_click` that indicates if that particular ad impression resulted in a click event. Note that our `input_fn` function returns a callable.

```
def input_fn(paths):
    names = ["domain", "hour", "device_type", "is_click"]
    record_defaults = [[""], [0], ["desktop"], [0]]    def _parse_csv(rows_string_tensor):
        columns = tf.decode_csv(rows_string_tensor, record_defaults)
        features = dict(zip(names, columns[:-1]))
        labels = columns[-1]
        return features, labels    def _input_fn():
        dataset = tf.data.TextLineDataset(paths)
        dataset = dataset.map(_parse_csv)
        dataset = dataset.batch(100)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels    return _input_fn
```

Now everything is setup and we can finally train our classifier.

```
paths = [<some CSVs to train our model on>]
estimator.train(input_fn=input_fn(paths), steps=None)
```

To export our trained model, we are using TensorFlow’s own SavedModel format. It’s important to give names to tensors and operations, to be able to address the specific parts during model inference.

```
columns = [('hour', tf.int64),
           ('domain', tf.string),
           ('device_type', tf.string)]
feature_placeholders = {
 name: tf.placeholder(dtype, [1], name=name + "_placeholder")
 for name, dtype in columns
}
export_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
    feature_placeholders)
estimator.export_savedmodel(model_dir, export_input_fn)
```

# Inspecting the exported model

The TensorFlow installation contains a tool called [saved_model_cli](https://www.tensorflow.org/programmers_guide/saved_model#cli_to_inspect_and_execute_savedmodel) to inspect a model. Let’s take a look:

```
$ saved_model_cli show --dir <model_export_dir> --allMetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:signature_def['predict']:
The given SavedModel SignatureDef contains the following input(s):
inputs['device_type'] tensor_info:
    dtype: DT_STRING
    shape: (-1)
    name: device_type_placeholder:0
inputs['domain'] tensor_info:
    dtype: DT_STRING
    shape: (-1)
    name: domain_placeholder:0
inputs['hour'] tensor_info:
    dtype: DT_INT64
    shape: (-1)
    name: hour_placeholder:0
The given SavedModel SignatureDef contains the following output(s):
outputs['class_ids'] tensor_info:
    dtype: DT_INT64
    shape: (-1, 1)
    name: linear/head/predictions/ExpandDims:0
outputs['classes'] tensor_info:
    dtype: DT_STRING
    shape: (-1, 1)
    name: linear/head/predictions/str_classes:0
outputs['logistic'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 1)
    name: linear/head/predictions/logistic:0
outputs['logits'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 1)
    name: linear/linear_model/weighted_sum:0
outputs['probabilities'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 2)
    name: linear/head/predictions/probabilities:0
Method name is: tensorflow/serving/predict
```

The important part is that this tells us how the input & output nodes are named, which features are part of the model and what data types are expected. For example, the output node providing probabilities of the current input data leading to a click, can be accessed by the name “linear/head/predictions/probabilities”. We will need this information for our Go implementation.

# Switching to Go

Unfortunately, TensorFlow’s Go bindings are hard to use and the documentation is rather sparse. We rely on the [tfgo](https://github.com/galeone/tfgo#tfgo-tensorflow-in-go) package to make our life a bit easier. First, we load the exported model. Then we create some sample input data and trigger the actual prediction. The placeholder names we use in the function `Exec` are taken from saved_model_cli tool output above.

```
// Import model.
model := tg.LoadModel("EXPORT", []string{"serve"}, &tf.SessionOptions{})// Create input tensors.
deviceInput, _ := tf.NewTensor([]string{"desktop"})
hourInput, _ := tf.NewTensor([]int64{19})
domainInput, _ := tf.NewTensor([]string{"example.com"})// Predict.
results := model.Exec(
 []tf.Output{
   model.Op("linear/head/predictions/probabilities", 0),
 }, map[tf.Output]*tf.Tensor{
   model.Op("device_type_placeholder", 0): deviceInput,
   model.Op("hour_placeholder", 0): hourInput,
   model.Op("domain_placeholder", 0): domainInput,
  ,
 )
predictions := results[0].Value().([][]float32)
```

That’s it! We have successfully imported our trained model and started predicting. In our example, we just passed one sample (someone visiting “example.com” on a desktop browser at 7pm), but we could also pass multiple samples in one go and do multiple predictions at the same time, which turned out quite useful in our case.

# Conclusions

Moving to TensorFlow allows us to improve our technical architecture as well as our machine learning capabilities. The provided example code shows you how to setup, train and export a machine learning model in Python. And then how to import and apply that model in Go. The full, executable source code is available at: https://github.com/travelaudience/tensorflow-export-import-example

The advantages of the setup with TensorFlow

- Eliminate duplicate code and potential inconsistencies between Python and Go implementation
- The general export/import flow is independent of the used machine learning algorithm and the used features. This gives us Data Scientists much more freedom to explore new algorithms and go crazy with our modelling. At the same time it shortens our development cycle and allows moving models to production faster.
- We can take advantage of all current and future things available in the TensorFlow ecosystem.

There are also some minor disadvantages

- Tensorflow’s Go binding are *not* covered by the TensorFlow API stability guarantees and may see huge changes or disappear completely.
- With TensorFlow, doing predictions is slower compared to a similar model using our old approach. We are still working on this.
