import Link from '@oracle/elements/Link';
import Panel from '@oracle/components/Panel';
import Spacing from '@oracle/elements/Spacing';
import Text from '@oracle/elements/Text';
import {
  UNITS_BETWEEN_SECTIONS
} from '@oracle/styles/units/spacing';
import Button from '@oracle/elements/Button';

export default function Setup() {
  return (
    <Spacing mb={UNITS_BETWEEN_SECTIONS}>
      <Panel>
        <Text warning>
          You need to configure an AI provider before you can generate pipelines using AI.
        </Text>

        <Spacing mt={1}>
          <Text warning>
            Go to <Link
              href="/settings/workspace/preferences"
              openNewWindow
            >
              Project Settings → AI provider
            </Link>{' '}
            and enter your API key (and optionally a custom Base URL and model name).
          </Text>
        </Spacing>

        <Spacing mt={1}>
          <Text muted small>
            Supported providers: <Text bold inline small>OpenAI</Text>,{' '}
            <Text bold inline small>Ollama</Text> (local),{' '}
            <Text bold inline small>Groq</Text>,{' '}
            <Text bold inline small>Together AI</Text>,{' '}
            <Text bold inline small>LM Studio</Text>, and any other OpenAI-compatible API.
          </Text>
        </Spacing>

        <Spacing mt={2}>
          <Text>
            Want to code faster and smarter with deeply integrated AI capabilities for building data pipelines?
          </Text>

          <Spacing mt={1}>
            <Button
              basic
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                window.open('https://www.mage.ai/ai?ref=oss', '_blank');
              }}
              style={{
                backgroundColor: '#00FB82',
                padding: '8px 16px',
                fontSize: 16,
                color: '#000000',
              }}
              pill
            >
              Try Mage Pro
            </Button>
          </Spacing>
        </Spacing>
      </Panel>
    </Spacing >
  )
}
