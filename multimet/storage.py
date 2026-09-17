# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Storage location handling shared by the gridded archive builders.

The builders accept a Zarr target that may be either a cloud URL or a path on
the local filesystem, and they have to decide which it is before choosing
between an ``fsspec`` mapper and a plain path. Getting that decision wrong is
expensive and confusing: a local path misread as a bucket name produces a URL
like ``gs://C:\\Users\\...`` that fails deep inside the storage stack rather
than at the call site.

This module keeps that classification in one place so both builders agree.
"""

from __future__ import annotations

import os
import re

# Matches a URI scheme prefix such as "gs://", "s3://" or "file://".
#
# A Windows drive letter cannot match this: "C:\\data" and "C:/data" have no
# "//" after the colon.
_URI_SCHEME_RE = re.compile(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*)://")

# Matches a Windows drive-qualified path such as "C:\\data" or "C:/data".
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")

_FILE_SCHEME_PREFIX = "file://"

# Errors that will fail identically on every attempt, so a retry loop should
# re-raise them immediately rather than sleeping between attempts. The usual
# case is a location whose protocol has no installed driver (for example a
# ``gs://`` URL without ``gcsfs``), which fsspec reports as an ImportError.
NON_RETRYABLE_ERRORS = (ImportError, TypeError)



def resolve_zarr_target(target: str) -> tuple[str, bool]:
  """Classifies a Zarr target as either remote or local.

  The rules, in order:

  1. An explicit URI scheme wins. ``file://`` is unwrapped to a bare local
     path; anything else (``gs://``, ``s3://``, ...) is remote.
  2. Anything that looks like a local filesystem path is local. This covers
     POSIX absolute paths, ``./`` and ``../`` prefixes, ``~``, Windows
     drive-qualified paths, UNC paths, and any path containing a backslash.
  3. Anything else is a bare ``bucket/key`` and is assumed to live in Google
     Cloud Storage, which is what the default targets look like.

  Args:
    target: A Zarr store location.

  Returns:
    A ``(location, is_remote)`` pair. When ``is_remote`` is true, ``location``
    is a fully qualified URL suitable for ``fsspec``. When it is false,
    ``location`` is a plain filesystem path.

  Raises:
    ValueError: If ``target`` is empty.
  """
  if not target:
    raise ValueError('Zarr target must be a non-empty string.')

  scheme_match = _URI_SCHEME_RE.match(target)
  if scheme_match:
    if scheme_match.group('scheme').lower() == 'file':
      return target[len(_FILE_SCHEME_PREFIX):], False
    return target, True

  looks_local = (
      _WINDOWS_DRIVE_RE.match(target) is not None
      # UNC share, e.g. \\server\share. Also caught by the backslash test
      # below, but spelled out because it is a genuinely distinct case.
      or target.startswith('\\\\')
      # Any backslash means a Windows-style path; bucket keys never use them.
      or '\\' in target
      or os.path.isabs(target)
      or target.startswith(('./', '../', '~'))
  )
  if looks_local:
    return target, False

  return f'gs://{target}', True


def is_remote_target(target: str) -> bool:
  """Returns whether ``target`` refers to a remote (cloud) store."""
  return resolve_zarr_target(target)[1]
