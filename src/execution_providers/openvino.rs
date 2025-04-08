use alloc::{ffi::CString, format};
use core::{ffi::c_void, ptr};

use crate::{
	error::{Error, Result},
	execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
	session::builder::SessionBuilder
};

#[derive(Debug, Clone)]
pub struct OpenVINOExecutionProvider {
	device_type: Option<CString>,
	precision: Option<CString>,
	device_id: Option<CString>,	// not use in v2
	num_threads: usize,
	num_streams: usize,
	cache_dir: Option<CString>,
	context: *mut c_void,
	enable_opencl_throttling: bool,
	enable_dynamic_shapes: bool,
	enable_npu_fast_compile: bool	// not use in v2
}

unsafe impl Send for OpenVINOExecutionProvider {}
unsafe impl Sync for OpenVINOExecutionProvider {}

impl Default for OpenVINOExecutionProvider {
	fn default() -> Self {
		Self {
			device_type: None,
			precision: None,
			device_id: None,
			num_threads: 8,
			num_streams: 1,
			cache_dir: None,
			context: ptr::null_mut(),
			enable_opencl_throttling: false,
			enable_dynamic_shapes: false,
			enable_npu_fast_compile: false
		}
	}
}

impl OpenVINOExecutionProvider {
	/// Overrides the accelerator hardware type and precision with these values at runtime. If this option is not
	/// explicitly set, default hardware and precision specified during build time is used.
	#[must_use]
	pub fn with_device_type(mut self, device_type: impl AsRef<str>) -> Self {
		self.device_type = Some(CString::new(device_type.as_ref()).expect("invalid string"));
		self
	}

	#[must_use]
	pub fn with_precision(mut self, precision: impl AsRef<str>) -> Self {
		self.precision = Some(CString::new(precision.as_ref()).expect("invalid string"));
		self
	}

	/// Selects a particular hardware device for inference. If this option is not explicitly set, an arbitrary free
	/// device will be automatically selected by OpenVINO runtime.
	#[must_use]
	pub fn with_device_id(mut self, device_id: impl AsRef<str>) -> Self {
		self.device_id = Some(CString::new(device_id.as_ref()).expect("invalid string"));
		self
	}

	/// Overrides the accelerator default value of number of threads with this value at runtime. If this option is not
	/// explicitly set, default value of 8 is used during build time.
	#[must_use]
	pub fn with_num_threads(mut self, num_threads: usize) -> Self {
		self.num_threads = num_threads;
		self
	}

	#[must_use]
	pub fn with_num_streams(mut self, num_streams: usize) -> Self {
		self.num_streams = num_streams;
		self
	}

	/// Explicitly specify the path to save and load the blobs, enabling model caching.
	#[must_use]
	pub fn with_cache_dir(mut self, dir: impl AsRef<str>) -> Self {
		self.cache_dir = Some(CString::new(dir.as_ref()).expect("invalid string"));
		self
	}

	/// This option is only alvailable when OpenVINO EP is built with OpenCL flags enabled. It takes in the remote
	/// context i.e the `cl_context` address as a void pointer.
	#[must_use]
	pub fn with_opencl_context(mut self, context: *mut c_void) -> Self {
		self.context = context;
		self
	}

	/// This option enables OpenCL queue throttling for GPU devices (reduces CPU utilization when using GPU).
	#[must_use]
	pub fn with_opencl_throttling(mut self, enable: bool) -> Self {
		self.enable_opencl_throttling = enable;
		self
	}

	/// This option if enabled works for dynamic shaped models whose shape will be set dynamically based on the infer
	/// input image/data shape at run time in CPU. This gives best result for running multiple inferences with varied
	/// shaped images/data.
	#[must_use]
	pub fn with_dynamic_shapes(mut self, enable: bool) -> Self {
		self.enable_dynamic_shapes = enable;
		self
	}

	#[must_use]
	pub fn with_npu_fast_compile(mut self, enable: bool) -> Self {
		self.enable_npu_fast_compile = enable;
		self
	}

	#[must_use]
	pub fn build(self) -> ExecutionProviderDispatch {
		self.into()
	}
}

impl From<OpenVINOExecutionProvider> for ExecutionProviderDispatch {
	fn from(value: OpenVINOExecutionProvider) -> Self {
		ExecutionProviderDispatch::new(value)
	}
}

impl ExecutionProvider for OpenVINOExecutionProvider {
	fn as_str(&self) -> &'static str {
		"OpenVINOExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(all(target_arch = "x86_64", any(target_os = "windows", target_os = "linux")))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "openvino"))]
		{
			use alloc::ffi::CString;
			use core::ffi::c_char;

			use crate::AsPointer;

			// Like TensorRT, the OpenVINO EP is also pretty picky about needing an environment by this point.
			let _ = crate::environment::get_environment();
/*
			let openvino_options = ort_sys::OrtOpenVINOProviderOptions {
				device_type: self
					.device_type
					.as_ref()
					.map_or_else(ptr::null, |x| x.as_bytes().as_ptr().cast::<c_char>()),
				device_id: self.device_id.as_ref().map_or_else(ptr::null, |x| x.as_bytes().as_ptr().cast::<c_char>()),
				num_of_threads: self.num_threads,
				cache_dir: self.cache_dir.as_ref().map_or_else(ptr::null, |x| x.as_bytes().as_ptr().cast::<c_char>()),
				context: self.context,
				enable_opencl_throttling: self.enable_opencl_throttling.into(),
				enable_dynamic_shapes: self.enable_dynamic_shapes.into(),
				enable_npu_fast_compile: self.enable_npu_fast_compile.into()
			};
			*/

			// use ort_sys::OrtOpenVINOProviderOptionsV2;

			let mut key_array: Vec<CString> = Vec::new();
			let mut value_array: Vec<CString> = Vec::new();
			if let Some(device_type) = &self.device_type {
				key_array.push(CString::new("device_type")?);
				value_array.push(device_type.clone());
			}

			if let Some(precision) = &self.precision {
				key_array.push(CString::new("precision")?);
				value_array.push(precision.clone());
			}

			key_array.push(CString::new("num_of_threads")?);
			value_array.push(CString::new(format!("{}", self.num_threads))?);

			key_array.push(CString::new("num_streams")?);
			value_array.push(CString::new(format!("{}", self.num_streams))?);

			if let Some(cache_dir) = &self.cache_dir {
				key_array.push(CString::new("cache_dir")?);
				value_array.push(cache_dir.clone());
			}

			if self.context != ptr::null_mut() {
				key_array.push(CString::new("context")?);
				value_array.push(CString::new(format!("{:p}", self.context))?);
			}

			if self.enable_opencl_throttling {
				key_array.push(CString::new("enable_opencl_throttling")?);
				value_array.push(CString::new("true")?);
			} else {
				key_array.push(CString::new("enable_opencl_throttling")?);
				value_array.push(CString::new("false")?);
			}

			if self.enable_dynamic_shapes {
				key_array.push(CString::new("disable_dynamic_shapes")?);
				value_array.push(CString::new("false")?);
			} else {
				key_array.push(CString::new("disable_dynamic_shapes")?);
				value_array.push(CString::new("true")?);
			}

			let param_size = key_array.len();
			let key_ptr_array = key_array
				.iter()
				.map(|x| x.as_ptr())
				.collect::<Vec<_>>();
			let key_ptr = key_ptr_array.as_ptr();
			let value_ptr_array = value_array
				.iter()
				.map(|x| x.as_ptr())
				.collect::<Vec<_>>();
			let value_ptr = value_ptr_array.as_ptr();

			// crate::ortsys![unsafe SessionOptionsAppendExecutionProvider_OpenVINO(session_builder.ptr_mut(), ptr::addr_of!(openvino_options))?];
			crate::ortsys![unsafe SessionOptionsAppendExecutionProvider_OpenVINO_V2(session_builder.ptr_mut(), key_ptr, value_ptr, param_size)?];
			return Ok(());
		}

		Err(Error::new(format!("`{}` was not registered because its corresponding Cargo feature is not enabled.", self.as_str())))
	}
}
